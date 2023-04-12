#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::cout << "usage: ./sfd_extract <sfd file> <symbol name>\n";
    return -1;
  }
  std::filesystem::path filepath = argv[1];
  std::string symbol_name = argv[2];
  std::ifstream file_stream(filepath);
  if (!file_stream.is_open()) {
    std::cout << "could not open " << filepath.string() << '\n';
    return -1;
  }
  std::string start_char_line = "StartChar: " + symbol_name;
  int line_number = 1;
  bool found_char = false;
  bool in_spline_set = false;
  for (std::string line; std::getline(file_stream, line); ++line_number) {
    if (found_char) {
      if (in_spline_set) {
        if (line == "EndSplineSet") {
          std::cout << "  }\n";
          break;
        }
        std::stringstream line_stream(line);
        if (line.find('m') != std::string::npos) {
          float x, y;
          line_stream >> x >> y;
          std::cout << "    moveto(" << x << ", " << y << ");\n";
        } else if (line.find('l') != std::string::npos) {
          float x, y;
          line_stream >> x >> y;
          std::cout << "    lineto(" << x << ", " << y << ");\n";
        } else if (line.find('c') != std::string::npos) {
          float x1, y1, x2, y2, x3, y3;
          line_stream >> x1 >> y1 >> x2 >> y2 >> x3 >> y3;
          std::cout << "    curveto(" << x1 << ", " << y1 << ", " << x3 << ", " << y3 << ");\n";
        }
      } else {
        if (line == "SplineSet") {
          in_spline_set = true;
          std::cout << "  void add_" << symbol_name << "() {\n";
        }
      }
    } else {
      if (line == start_char_line) {
        found_char = true;
      }
    }
  }
  file_stream.close();
}
