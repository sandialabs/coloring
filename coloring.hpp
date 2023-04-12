#pragma once

#include <filesystem>
#include <cmath>
#include <vector>
#include <optional>
#include <string>

namespace coloring {

inline constexpr float pi = 3.14159265358979323846f;

inline constexpr float
square(float a)
{
  return a * a;
}

inline constexpr float
cube(float a)
{
  return a * a * a;
}

inline constexpr int
min(int a, int b)
{
  return (b < a) ? b : a;
}

inline constexpr int
max(int a, int b)
{
  return (b > a) ? b : a;
}

inline constexpr float
min(float a, float b)
{
  return (b < a) ? b : a;
}

inline constexpr float
max(float a, float b)
{
  return (b > a) ? b : a;
}

inline constexpr float
clamp(float v, float lo, float hi )
{
  return min(max(v, lo), hi);
}

class float_rgb;

class byte_rgb {
  unsigned char m_red{0};
  unsigned char m_green{0};
  unsigned char m_blue{0};
 public:
  constexpr byte_rgb() = default;
  constexpr explicit byte_rgb(float_rgb const&);
  constexpr byte_rgb(
      unsigned char red_arg,
      unsigned char green_arg,
      unsigned char blue_arg)
    :m_red(red_arg)
    ,m_green(green_arg)
    ,m_blue(blue_arg)
  {
  }
  constexpr unsigned char r() const { return m_red; }
  constexpr unsigned char g() const { return m_green; }
  constexpr unsigned char b() const { return m_blue; }
  void store(unsigned char* bytes) const
  {
    bytes[0] = m_red;
    bytes[1] = m_green;
    bytes[2] = m_blue;
  }
  static byte_rgb load(unsigned char const* bytes)
  {
    byte_rgb result;
    result.m_red = bytes[0];
    result.m_green = bytes[1];
    result.m_blue = bytes[2];
    return result;
  }
};

inline constexpr byte_rgb black = byte_rgb(0, 0, 0);
inline constexpr byte_rgb white = byte_rgb(255, 255, 255);
inline constexpr byte_rgb red = byte_rgb(255, 0, 0);
inline constexpr byte_rgb green = byte_rgb(0, 255, 0);
inline constexpr byte_rgb blue = byte_rgb(0, 0, 255);

namespace okabe_ito {
  // a palette of 8 colors (when black is added) that is supposed to be friendly
  // to people with color blindness
  // https://jfly.uni-koeln.de/color/#select
  inline constexpr byte_rgb orange = byte_rgb(230, 159, 0);
  inline constexpr byte_rgb sky_blue = byte_rgb(86, 180, 233);
  inline constexpr byte_rgb bluish_green = byte_rgb(0, 158, 115);
  inline constexpr byte_rgb yellow = byte_rgb(240, 228, 66);
  inline constexpr byte_rgb blue = byte_rgb(0, 114, 178);
  inline constexpr byte_rgb vermillion = byte_rgb(213, 94, 0);
  inline constexpr byte_rgb reddish_purple = byte_rgb(204, 121, 167);
  static const byte_rgb colors[7] = {
    orange,
    sky_blue,
    bluish_green,
    yellow,
    blue,
    vermillion,
    reddish_purple
  };
}

inline constexpr unsigned char ftob(float x)
{
  return clamp(x, 0.0f, 1.0f) * 255;
}

inline constexpr float btof(unsigned char x)
{
  return float(x) / 255.0f;
}

class float_rgba;

class byte_rgba {
  byte_rgb m_rgb;
  unsigned char m_alpha{0};
 public:
  constexpr byte_rgba() = default;
  constexpr byte_rgba(
      byte_rgb const& rgb_arg,
      float alpha_arg)
    :m_rgb(rgb_arg)
    ,m_alpha(ftob(alpha_arg))
  {
  }
  constexpr byte_rgba(byte_rgb const& rgb_arg)
    :byte_rgba(rgb_arg, 1.0f)
  {
  }
  constexpr explicit byte_rgba(float_rgba const&);
  constexpr byte_rgb const& rgb() const { return m_rgb; }
  constexpr unsigned char a() const { return m_alpha; }
  void store(unsigned char* mem) const
  {
    m_rgb.store(mem);
    mem[3] = m_alpha;
  }
  static byte_rgba load(unsigned char const* mem)
  {
    byte_rgba result;
    result.m_rgb = byte_rgb::load(mem);
    result.m_alpha = mem[3];
    return result;
  }
  constexpr byte_rgba over(byte_rgba const&) const;
};

class float_rgb {
  float m_red{std::numeric_limits<float>::quiet_NaN()};
  float m_green{std::numeric_limits<float>::quiet_NaN()};
  float m_blue{std::numeric_limits<float>::quiet_NaN()};
 public:
  constexpr float_rgb() = default;
  constexpr explicit float_rgb(byte_rgb const& b)
    :m_red(btof(b.r()))
    ,m_green(btof(b.g()))
    ,m_blue(btof(b.b()))
  {
  }
  constexpr float_rgb(
      float r_arg,
      float g_arg,
      float b_arg)
    :m_red(r_arg)
    ,m_green(g_arg)
    ,m_blue(b_arg)
  {
  }
  constexpr float r() const { return m_red; }
  constexpr float g() const { return m_green; }
  constexpr float b() const { return m_blue; }
};

inline constexpr float_rgb
operator*(float_rgb const& a, float b)
{
  return float_rgb(
      a.r() * b,
      a.g() * b,
      a.b() * b);
}

inline constexpr float_rgb
operator/(float_rgb const& a, float b)
{
  float const inverse = 1.0f / b;
  return float_rgb(
      a.r() * inverse,
      a.g() * inverse,
      a.b() * inverse);
}

inline constexpr float_rgb
operator+(float_rgb const& a, float_rgb const& b)
{
  return float_rgb(
      a.r() + b.r(),
      a.g() + b.g(),
      a.b() + b.b());
}

constexpr byte_rgb::byte_rgb(float_rgb const& f)
  :m_red(ftob(f.r()))
  ,m_green(ftob(f.g()))
  ,m_blue(ftob(f.b()))
{
}

class float_rgba {
  float_rgb m_rgb;
  float m_alpha{std::numeric_limits<float>::quiet_NaN()};
 public:
  constexpr float_rgba() = default;
  constexpr float_rgba(byte_rgba const& rgba_arg)
    :m_rgb(rgba_arg.rgb())
    ,m_alpha(btof(rgba_arg.a()))
  {
  }
  constexpr float_rgba(
      float_rgb const& rgb_arg,
      float const& a_arg)
    :m_rgb(rgb_arg)
    ,m_alpha(a_arg)
  {
  }
  constexpr float_rgb const& rgb() const { return m_rgb; }
  constexpr float a() const { return m_alpha; }
  // https://en.wikipedia.org/wiki/Alpha_compositing#Description
  constexpr float_rgba over(float_rgba const& background) const
  {
    float_rgba result;
    float const background_weight = background.a() * (1.0f - a());
    result.m_alpha = a() + background_weight;
    result.m_rgb =
      (rgb() * a() + background.rgb() * background_weight)
      / result.m_alpha;
    return result;
  }
};

constexpr byte_rgba::byte_rgba(float_rgba const& f)
  :m_rgb(f.rgb())
  ,m_alpha(ftob(f.a()))
{
}

constexpr byte_rgba byte_rgba::over(byte_rgba const& fore) const
{
  return byte_rgba(float_rgba(*this).over(float_rgba(fore)));
}

enum plot_style {
  lines,
  points
};

class plot_dataset {
 public:
  plot_style style{lines};
  std::optional<float> thickness;
  std::vector<float> x;
  std::vector<float> y;
};

class plot_group {
 public:
  std::string name;
  std::optional<byte_rgba> color;
  std::vector<plot_dataset> datasets;
};

enum class legend_location {
  no_legend,
  upper_left,
  upper_right,
  lower_left,
  lower_right
};

enum class axis_scale {
  linear,
  log,
  symlog
};

class plot_data {
 public:
  std::string title;
  std::string xlabel;
  std::string ylabel;
  axis_scale xscale{axis_scale::linear};
  axis_scale yscale{axis_scale::linear};
  coloring::legend_location legend_location{
    coloring::legend_location::no_legend};
  std::vector<plot_group> groups;
};

void plot(
    std::filesystem::path const& pngpath,
    plot_data& data,
    unsigned width = 640,
    unsigned height = 480);

}
