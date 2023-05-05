import argparse
import wincnn
import io

from sympy import Rational, symbols, Matrix, Poly, zeros, eye, Indexed, simplify, IndexedBase, init_printing, pprint, count_ops
from sympy.simplify.radsimp import collect_const


parser = argparse.ArgumentParser(description="Winograd convolution")

parser.add_argument(
    "--input-width",
    default=2,
    type=int,
    help="Input width (and height)",
)

parser.add_argument(
    "--padding",
    default=1,
    type=int,
    help="Padding width to input",
)

parser.add_argument(
    "--kernel-width",
    default=3,
    type=int,
    help="Kernel width (and height)",
)

parser.add_argument(
    "--in-channel",
    default=128,
    type=int,
    help="Number of input channels",
)

parser.add_argument(
    "--out-channel",
    default=128,
    type=int,
    help="Number of output channels",
)

parser.add_argument(
    "--batch",
    default=128,
    type=int,
    help="Number of batches",
)


# Ref to https://github.com/andravin/wincnn/issues/24
interpolation_points = [
    0, 1, -1, 2, -2, Rational(1,2), -Rational(1,2),
    Rational(3,2), -Rational(3,2), Rational(2,3), -Rational(2,3),
    Rational(2,5), -Rational(2,5), Rational(5,2), -Rational(5,2)
]

def genConv2dMatrics(kernel_width, input_width, padding,
        input_name="d", kernel_name="g", intermediate_name="m"):
    n = input_width + padding * 2
    r = kernel_width
    m = n - r + 1

    di = IndexedBase(input_name)
    d = Matrix(n, n, lambda i,j: di[(i - padding) * input_width + j - padding] if \
        i >= padding and i < input_width + padding
        and j >= padding and j < input_width + padding \
        else 0)

    gi = IndexedBase(kernel_name)
    g = Matrix(r, r, lambda i,j: gi[i * r + j])
    
    print ("d = ")
    pprint(d)
    print ("")

    print ("g = ")
    pprint(g)
    print ("")

    AT, G, BT, f = wincnn.cookToomFilter(interpolation_points[0:m+r-2],m,r,wincnn.FractionsInG)

    print ("A = ")
    pprint(AT.transpose())
    print ("")

    print ("G = ")
    pprint(G)
    print ("")

    print ("B = ")
    pprint(BT.transpose())
    print ("")

    U = simplify(G * g * G.transpose())
    print ("U = ")
    pprint(U)
    print ("")

    V = simplify(BT * d * BT.transpose())
    print ("V = ")
    pprint(V)
    print ("")

    mi = IndexedBase(intermediate_name)
    M = Matrix(m + r - 1, m + r - 1, lambda i,j: mi[i * (m + r - 1) + j])
    print ("M = ")
    pprint(M)
    print ("")

    Y = simplify(AT * M * AT.transpose())
    print ("Y = ")
    pprint(Y)
    print ("")

    print(collect_const(U[5]))
    return U, V, Y

def genConv2dCode(U, V, Y, kernel_width, input_width, padding, in_channel, out_channel, batch,
                  input_name="d", kernel_name="g", intermediate_name="m", data_type_name="double"):
    n = input_width + padding * 2
    r = kernel_width
    m = n - r + 1
    m_size = (m + r - 1) * (m + r - 1)

    code_io = io.StringIO()
    code_io.write(
        "static void winograd_conv2d_nchw_{}_{}_{}_{}_{}(const double *input, const double *filter, double *output) {{\n".format(
            batch, in_channel, input_width, kernel_width, padding
        )
    )
    code_io.write("\tfor (int rc = 0; rc < {}; ++rc) {{  // in_channel\n".format(in_channel))
    code_io.write("\t\t{} V[{}][{}];\n".format(
        data_type_name, batch, m_size))
    code_io.write("\t\tfor (int nn = 0; nn < {}; ++nn) {{  // batch\n".format(batch))
    code_io.write("\t\t\tconst {} *{} = &input[((nn * {}) + rc) * {}];  // f(nn, rc)\n".format(
        data_type_name, input_name, in_channel, input_width * input_width))
    for i in range(m_size):
        code_io.write("\t\t\tV[nn][{}] = {};\n".format(i, collect_const(V[i])))
    code_io.write("\t\t}\n")
    code_io.write("\t\tfor (int ff = 0; ff < {}; ++ff) {{  // out_channel\n".format(out_channel))
    code_io.write("\t\t\tconst {} *{} = &filter[((ff * {}) + rc) * {}];  // f(rc, ff) \n".format(
        data_type_name, kernel_name, in_channel, kernel_width * kernel_width))
    code_io.write("\t\t\t{} U[{}];\n".format(data_type_name, m_size))
    for i in range(m_size):
        code_io.write("\t\t\tU[{}] = {};\n".format(i, collect_const(U[i])))
    code_io.write("\t\t\tfor (int nn = 0; nn < {}; ++nn) {{  // batch\n".format(batch))
    code_io.write("\t\t\t\t{} {}[{}];\n".format(data_type_name, intermediate_name, m_size))
    for i in range(m_size):
        code_io.write("\t\t\t\t{}[{}] = U[{}] * V[nn][{}];\n".format(intermediate_name, i, i, i))
    code_io.write("\t\t\t\t{} *out = &output[((nn * {}) + ff) * {}];  // f(nn, ff)\n".format(
        data_type_name, out_channel, m * m))
    for i in range(m * m):
        code_io.write("\t\t\t\tout[{}] += {};\n".format(i, Y[i]))
    code_io.write("\t\t\t}\n")
    code_io.write("\t\t}\n")
    code_io.write("\t}\n")
    code_io.write("}\n")

    code_str = code_io.getvalue()
    code_io.close()
    return code_str


if __name__ == '__main__':
    args = parser.parse_args()

    U, V, Y = genConv2dMatrics(args.kernel_width, args.input_width, args.padding)
    code_str = genConv2dCode(U, V, Y, args.kernel_width, args.input_width, args.padding,
                             args.in_channel, args.out_channel, args.batch)
    print(code_str)
