import math
from pathlib import Path

BOARD_CENTER = 4.5
WEIGHT_LOG_BASE = 9.0
PENALTY_LOG_BASE = 2.13157
MIN_QUADRANT_SUM = 1e-6
SCALE = 256  # penalties stored in centipawns * SCALE

# Directions order: up, down, left, right
DIRECTIONS = [
    (0, 1),
    (0, -1),
    (-1, 0),
    (1, 0),
]

MAX_LEN = 7

prefix_squares = {}
for sq in range(64):
    file = sq % 8
    rank = sq // 8
    per_dir = []
    for df, dr in DIRECTIONS:
        squares = []
        f, r = file, rank
        while True:
            f += df
            r += dr
            if not (0 <= f < 8 and 0 <= r < 8):
                break
            squares.append((f, r))
        per_dir.append(squares)
    prefix_squares[sq] = per_dir


def dir_sums(squares, count):
    sx = sy = 0.0
    limit = min(count, len(squares))
    for i in range(limit):
        f, r = squares[i]
        x = abs((f + 1) - BOARD_CENTER)
        y = abs((r + 1) - BOARD_CENTER)
        sx += x
        sy += y
    return sx, sy


def quadrant_term(sums_a, count_a, sums_b, count_b):
    if count_a == 0 or count_b == 0:
        return 0.0
    x1 = sums_a[0] / count_a
    y1 = sums_a[1] / count_a
    x2 = sums_b[0] / count_b
    y2 = sums_b[1] / count_b
    xbar = 0.5 * (x1 + x2)
    ybar = 0.5 * (y1 + y2)
    term = 2.0 * (xbar * xbar + ybar * ybar)
    if term <= 0.0:
        return 0.0
    weight = 2.0 - 0.5 * (math.log(term) / math.log(WEIGHT_LOG_BASE))
    return count_a * count_b * weight


def penalty_for_counts(sq, up, down, left, right):
    directions = prefix_squares[sq]
    sums = [
        [dir_sums(directions[i], k) for k in range(MAX_LEN + 1)]
        for i in range(4)
    ]

    total = 0.0
    total += quadrant_term(sums[0][up], up, sums[3][right], right)
    total += quadrant_term(sums[0][up], up, sums[2][left], left)
    total += quadrant_term(sums[1][down], down, sums[3][right], right)
    total += quadrant_term(sums[1][down], down, sums[2][left], left)

    if total <= 0.0:
        penalty = 4.0 - math.log(MIN_QUADRANT_SUM) / math.log(PENALTY_LOG_BASE)
    else:
        penalty = 4.0 - (math.log(total) / math.log(PENALTY_LOG_BASE))

    scaled = round(penalty * SCALE)
    return max(-32768, min(32767, scaled))


def generate_table():
    table = []
    for sq in range(64):
        entry = []
        for up in range(MAX_LEN + 1):
            up_slice = []
            for down in range(MAX_LEN + 1):
                down_slice = []
                for left in range(MAX_LEN + 1):
                    row = []
                    for right in range(MAX_LEN + 1):
                        row.append(penalty_for_counts(sq, up, down, left, right))
                    down_slice.append(row)
                up_slice.append(down_slice)
            entry.append(up_slice)
        table.append(entry)
    return table


def emit_table(table):
    lines = []
    lines.append("#ifndef ROOK_ACTIVITY_TABLE_H_INCLUDED")
    lines.append("#define ROOK_ACTIVITY_TABLE_H_INCLUDED")
    lines.append("")
    lines.append("#include <array>")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("namespace Stockfish {")
    lines.append("")
    lines.append("constexpr int RookPenaltyScale = {}; // penalties stored in centipawns * scale".format(SCALE))
    lines.append("")
    len_str = str(MAX_LEN + 1)
    lines.append(
        "inline constexpr std::array<std::array<std::array<std::array<std::array<int16_t, {0}>, {0}>, {0}>, {0}>, 64> RookPenaltyTable = {{".format(
            len_str, len_str, len_str, len_str
        )
    )
    for sq, entry in enumerate(table):
        lines.append("    {")
        for up_idx, up_slice in enumerate(entry):
            lines.append("        {")
            for down_idx, down_slice in enumerate(up_slice):
                row_strings = []
                for left_slice in down_slice:
                    row_strings.append("{" + ", ".join(str(v) for v in left_slice) + "}")
                lines.append("            {" + ", ".join(row_strings) + "}" + ("," if down_idx != MAX_LEN else ""))
            lines.append("        }" + ("," if up_idx != MAX_LEN else ""))
        lines.append("    }" + ("," if sq != 63 else ""))
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace Stockfish")
    lines.append("")
    lines.append("#endif  // ROOK_ACTIVITY_TABLE_H_INCLUDED")
    return "\n".join(lines)


def main():
    table = generate_table()
    header = emit_table(table)
    path = Path(__file__).resolve().parents[1] / "src" / "rook_activity_table.h"
    path.write_text(header)


if __name__ == "__main__":
    main()
