# python/convert_tflite_to_header.py

import sys
from pathlib import Path


def convert_tflite_to_header(tflite_path: str, header_path: str, var_name: str):
    """
    Convert a .tflite file to a C header with a byte array:
        const unsigned char <var_name>[] = {...};
        const int <var_name>_len = N;
    """
    tflite_bytes = Path(tflite_path).read_bytes()
    n = len(tflite_bytes)

    with open(header_path, "w") as f:
        f.write("// Auto-generated from %s\n" % tflite_path)
        f.write("#include <cstdint>\n\n")
        f.write("const unsigned char %s[] = {\n  " % var_name)

        for i, b in enumerate(tflite_bytes):
            f.write(str(b))
            if i != n - 1:
                f.write(",")
            if (i + 1) % 20 == 0:
                f.write("\n  ")

        f.write("\n};\n")
        f.write("const int %s_len = %d;\n" % (var_name, n))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_tflite_to_header.py model.tflite out.h var_name")
        sys.exit(1)
    convert_tflite_to_header(sys.argv[1], sys.argv[2], sys.argv[3])
