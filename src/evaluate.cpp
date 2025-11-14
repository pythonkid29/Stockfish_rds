/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "evaluate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <tuple>

#include <array>
#include <utility>

#include "nnue/network.h"
#include "nnue/nnue_misc.h"
#include "rook_activity_table.h"
#include "position.h"
#include "types.h"
#include "uci.h"
#include "nnue/nnue_accumulator.h"

namespace Stockfish {

namespace {
constexpr int DirectionCount = 4;

struct DirectionOffset {
    int df;
    int dr;
};

constexpr std::array<DirectionOffset, DirectionCount> DirectionOffsets = {{{0, 1}, {0, -1}, {-1, 0}, {1, 0}}};

int gather_direction_count(const Position& pos, Square sq, Color color, const DirectionOffset& offset) {
    int file = int(file_of(sq));
    int rank = int(rank_of(sq));
    int count = 0;

    while (true)
    {
        file += offset.df;
        rank += offset.dr;

        if (file < 0 || file >= 8 || rank < 0 || rank >= 8)
            break;

        Square target = make_square(File(file), Rank(rank));
        Piece  piece  = pos.piece_on(target);

        if (piece == NO_PIECE)
        {
            ++count;
            continue;
        }

        if (color_of(piece) == color)
            break;

        ++count;  // Capture square
        break;
    }

    return count;
}

int rook_penalty_scaled_for_square(const Position& pos, Square sq, Color color) {
    const int up    = gather_direction_count(pos, sq, color, DirectionOffsets[0]);
    const int down  = gather_direction_count(pos, sq, color, DirectionOffsets[1]);
    const int left  = gather_direction_count(pos, sq, color, DirectionOffsets[2]);
    const int right = gather_direction_count(pos, sq, color, DirectionOffsets[3]);

    return RookPenaltyTable[sq][up][down][left][right];
}

int rook_penalty_scaled(const Position& pos, Color c) {
    Bitboard rooks = pos.pieces(c, ROOK);
    int      total = 0;

    while (rooks)
    {
        Square sq = pop_lsb(rooks);
        total += rook_penalty_scaled_for_square(pos, sq, c);
    }

    return total;
}

double logistic_scale_a(const Position& pos) {
    const int material = pos.count<PAWN>() + 3 * pos.count<KNIGHT>() + 3 * pos.count<BISHOP>()
                       + 5 * pos.count<ROOK>() + 9 * pos.count<QUEEN>();

    const double m = std::clamp(material, 17, 78) / 58.0;

    constexpr std::array<double, 4> as = {-13.50030198, 40.92780883, -36.82753545, 386.83004070};

    return ((as[0] * m + as[1]) * m + as[2]) * m + as[3];
}

Value scaled_cp_to_value(int cpScaled, const Position& pos) {
    if (cpScaled == 0)
        return VALUE_ZERO;

    const double a = logistic_scale_a(pos);
    return Value(std::lround(cpScaled * a / (100.0 * RookPenaltyScale)));
}

}  // namespace

// Returns a static, purely materialistic evaluation of the position from
// the point of view of the side to move. It can be divided by PawnValue to get
// an approximation of the material advantage on the board in terms of pawns.
int Eval::simple_eval(const Position& pos) {
    Color c = pos.side_to_move();
    return PawnValue * (pos.count<PAWN>(c) - pos.count<PAWN>(~c))
         + (pos.non_pawn_material(c) - pos.non_pawn_material(~c));
}

bool Eval::use_smallnet(const Position& pos) { return std::abs(simple_eval(pos)) > 962; }

// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Eval::NNUE::Networks&    networks,
                     const Position&                pos,
                     Eval::NNUE::AccumulatorStack&  accumulators,
                     Eval::NNUE::AccumulatorCaches& caches,
                     int                            optimism) {

    assert(!pos.checkers());

    bool smallNet           = use_smallnet(pos);
    auto [psqt, positional] = smallNet ? networks.small.evaluate(pos, accumulators, &caches.small)
                                       : networks.big.evaluate(pos, accumulators, &caches.big);

    Value nnue = (125 * psqt + 131 * positional) / 128;

    // Re-evaluate the position when higher eval accuracy is worth the time spent
    if (smallNet && (std::abs(nnue) < 236))
    {
        std::tie(psqt, positional) = networks.big.evaluate(pos, accumulators, &caches.big);
        nnue                       = (125 * psqt + 131 * positional) / 128;
        smallNet                   = false;
    }

    // Blend optimism and eval with nnue complexity
    int nnueComplexity = std::abs(psqt - positional);
    optimism += optimism * nnueComplexity / 468;
    nnue -= nnue * nnueComplexity / 18000;

    int material = 535 * pos.count<PAWN>() + pos.non_pawn_material();
    int v        = (nnue * (77777 + material) + optimism * (7777 + material)) / 77777;

    const int whitePenaltyScaled = rook_penalty_scaled(pos, WHITE);
    const int blackPenaltyScaled = rook_penalty_scaled(pos, BLACK);
    const int cpAdjustScaled     = blackPenaltyScaled - whitePenaltyScaled;

    if (cpAdjustScaled)
        v += scaled_cp_to_value(cpAdjustScaled, pos);

    // Damp down the evaluation linearly when shuffling
    v -= v * pos.rule50_count() / 212;

    // Guarantee evaluation does not hit the tablebase range
    v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

    return v;
}

// Like evaluate(), but instead of returning a value, it returns
// a string (suitable for outputting to stdout) that contains the detailed
// descriptions and values of each evaluation term. Useful for debugging.
// Trace scores are from white's point of view
std::string Eval::trace(Position& pos, const Eval::NNUE::Networks& networks) {

    if (pos.checkers())
        return "Final evaluation: none (in check)";

    auto accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();
    auto caches       = std::make_unique<Eval::NNUE::AccumulatorCaches>(networks);

    std::stringstream ss;
    ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
    ss << '\n' << NNUE::trace(pos, networks, *caches) << '\n';

    ss << std::showpoint << std::showpos << std::fixed << std::setprecision(2) << std::setw(15);

    auto [psqt, positional] = networks.big.evaluate(pos, *accumulators, &caches->big);
    Value v                 = psqt + positional;
    v                       = pos.side_to_move() == WHITE ? v : -v;
    ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)\n";

    v = evaluate(networks, pos, *accumulators, *caches, VALUE_ZERO);
    v = pos.side_to_move() == WHITE ? v : -v;
    ss << "Final evaluation       " << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)";
    ss << " [with scaled NNUE, ...]";
    ss << "\n";

    return ss.str();
}

}  // namespace Stockfish
