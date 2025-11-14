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
#include "position.h"
#include "types.h"
#include "uci.h"
#include "nnue/nnue_accumulator.h"

namespace Stockfish {

namespace {

constexpr double BoardCenter = 4.5;
constexpr double WeightLogBase = 9.0;
constexpr double PenaltyLogBase = 2.13157;
constexpr double MinQuadrantSum = 1e-6;

struct DirectionData {
    int    count = 0;
    double xSum  = 0.0;
    double ySum  = 0.0;

    void add(int fileIndex, int rankIndex) {
        ++count;
        const double x = std::abs((fileIndex + 1) - BoardCenter);
        const double y = std::abs((rankIndex + 1) - BoardCenter);
        xSum += x;
        ySum += y;
    }

    std::pair<double, double> averages() const {
        if (!count)
            return {0.0, 0.0};
        return {xSum / count, ySum / count};
    }
};

DirectionData gather_direction(const Position& pos, Square sq, int df, int dr) {
    DirectionData data;
    int           file = int(file_of(sq));
    int           rank = int(rank_of(sq));

    while (true)
    {
        file += df;
        rank += dr;

        if (file < 0 || file >= 8 || rank < 0 || rank >= 8)
            break;

        Square target = make_square(File(file), Rank(rank));
        data.add(file, rank);

        if (pos.piece_on(target) != NO_PIECE)
            break;
    }

    return data;
}

double directional_weight(double xbar, double ybar) {
    const double term = 2.0 * (xbar * xbar + ybar * ybar);
    if (term <= 0.0)
        return 0.0;
    static const double logBase = std::log(WeightLogBase);
    return 2.0 - 0.5 * (std::log(term) / logBase);
}

double combine_quadrant(const DirectionData& first, const DirectionData& second) {
    if (!first.count || !second.count)
        return 0.0;

    const auto [x1, y1] = first.averages();
    const auto [x2, y2] = second.averages();
    const double xbar   = (x1 + x2) * 0.5;
    const double ybar   = (y1 + y2) * 0.5;
    const double weight = directional_weight(xbar, ybar);

    return (first.count * second.count) * weight;
}

double rook_penalty_cp_for_square(const Position& pos, Square sq) {
    const DirectionData up    = gather_direction(pos, sq, 0, 1);
    const DirectionData down  = gather_direction(pos, sq, 0, -1);
    const DirectionData right = gather_direction(pos, sq, 1, 0);
    const DirectionData left  = gather_direction(pos, sq, -1, 0);

    const double quadrantSum = combine_quadrant(up, right) + combine_quadrant(up, left)
                             + combine_quadrant(down, right) + combine_quadrant(down, left);

    static const double logBase     = std::log(PenaltyLogBase);
    static const double penaltyCap  = 4.0 - std::log(MinQuadrantSum) / logBase;

    if (quadrantSum <= 0.0)
        return penaltyCap;

    return 4.0 - (std::log(quadrantSum) / logBase);
}

double rook_penalty_cp(const Position& pos, Color c) {
    Bitboard rooks = pos.pieces(c, ROOK);
    double   total = 0.0;

    while (rooks)
    {
        Square sq = pop_lsb(rooks);
        total += rook_penalty_cp_for_square(pos, sq);
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

Value cp_to_value(double cp, const Position& pos) {
    if (cp == 0.0)
        return VALUE_ZERO;

    const double a = logistic_scale_a(pos);
    return Value(std::lround(cp * a / 100.0));
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

    const double whitePenaltyCp = rook_penalty_cp(pos, WHITE);
    const double blackPenaltyCp = rook_penalty_cp(pos, BLACK);
    const double cpAdjust       = blackPenaltyCp - whitePenaltyCp;

    if (cpAdjust != 0.0)
        v += cp_to_value(cpAdjust, pos);

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
