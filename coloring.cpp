#include "coloring.hpp"

#include "lodepng.h"

namespace coloring {

class int_vec {
  int m_x{0};
  int m_y{0};
 public:
  constexpr int_vec() = default;
  constexpr int_vec(int x_arg, int y_arg)
    :m_x(x_arg)
    ,m_y(y_arg)
  {
  }
  constexpr int x() const { return m_x; }
  constexpr int y() const { return m_y; }
  constexpr int& x() { return m_x; }
  constexpr int& y() { return m_y; }
};

class float_vec {
  float m_x{std::numeric_limits<float>::quiet_NaN()};
  float m_y{std::numeric_limits<float>::quiet_NaN()};
 public:
  constexpr float_vec() = default;
  constexpr float_vec(float x_arg, float y_arg)
    :m_x(x_arg)
    ,m_y(y_arg)
  {
  }
  constexpr float x() const { return m_x; }
  constexpr float y() const { return m_y; }
  constexpr float& x() { return m_x; }
  constexpr float& y() { return m_y; }
};

inline constexpr float_vec
operator-(float_vec const& a)
{
  return float_vec(-a.x(), -a.y());
}

inline constexpr float_vec
operator-(float_vec const& a, float_vec const& b)
{
  return float_vec(a.x() - b.x(), a.y() - b.y());
}

inline constexpr float_vec
operator+(float_vec const& a, float_vec const& b)
{
  return float_vec(a.x() + b.x(), a.y() + b.y());
}

inline constexpr float_vec
operator*(float_vec const& a, float b)
{
  return float_vec(a.x() * b, a.y() * b);
}

inline constexpr float_vec
operator*(float a, float_vec const& b)
{
  return float_vec(a * b.x(), a * b.y());
}

inline constexpr float_vec
operator/(float_vec const& a, float b)
{
  return float_vec(a.x() / b, a.y() / b);
}

inline constexpr float
dot(float_vec const& a, float_vec const& b)
{
  return a.x() * b.x() + a.y() * b.y();
}

inline constexpr float
squared_magnitude(float_vec const& a)
{
  return dot(a, a);
}

inline constexpr float
magnitude(float_vec const& a)
{
  return std::sqrt(squared_magnitude(a));
}

inline constexpr float_vec
normalize(float_vec const& a)
{
  return a / magnitude(a);
}

inline constexpr bool
operator==(float_vec const& a, float_vec const& b)
{
  return a.x() == b.x() && a.y() == b.y();
}

inline constexpr float_vec
rotate_left_90deg(float_vec const& a)
{
  return float_vec(-a.y(), a.x());
}

inline constexpr float_vec
rotate_right_90deg(float_vec const& a)
{
  return float_vec(a.y(), -a.x());
}

class circle {
  float_vec m_center;
  float m_radius_squared{0.0f};
 public:
  constexpr circle() = default;
  constexpr circle(
      float_vec const& center_arg,
      float radius_squared_arg)
    :m_center(center_arg)
    ,m_radius_squared(radius_squared_arg)
  {
  }
  constexpr float_vec const& center() const { return m_center; }
  constexpr float radius_squared() const { return m_radius_squared; }
};

class float_box {
  float_vec m_lower{
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max()
  };
  float_vec m_upper{
    std::numeric_limits<float>::lowest(),
    std::numeric_limits<float>::lowest()
  };
 public:
  constexpr float_box() = default;
  constexpr float_box(
      float_vec const& lower_arg,
      float_vec const& upper_arg)
    :m_lower(lower_arg)
    ,m_upper(upper_arg)
  {
  }
  constexpr float_vec const& lower() const { return m_lower; }
  constexpr float_vec const& upper() const { return m_upper; }
  constexpr circle bounding_circle() const
  {
    auto const half_diagonal = (m_upper - m_lower) * 0.5f;
    auto const center = m_lower + half_diagonal;
    auto const radius_squared = squared_magnitude(half_diagonal);
    return circle(center, radius_squared);
  }
  constexpr void include(float_vec const& point)
  {
    m_lower.x() = min(m_lower.x(), point.x());
    m_lower.y() = min(m_lower.y(), point.y());
    m_upper.x() = max(m_upper.x(), point.x());
    m_upper.y() = max(m_upper.y(), point.y());
  }
  constexpr void include(float_box const& box)
  {
    m_lower.x() = min(m_lower.x(), box.lower().x());
    m_lower.y() = min(m_lower.y(), box.lower().y());
    m_upper.x() = max(m_upper.x(), box.upper().x());
    m_upper.y() = max(m_upper.y(), box.upper().y());
  }
  constexpr float_vec extents() const
  {
    return float_vec(m_upper.x() - m_lower.x(), m_upper.y() - m_lower.y());
  }
  constexpr float_vec upper_left() const
  {
    return float_vec(m_lower.x(), m_upper.y());
  }
  constexpr float_vec upper_right() const
  {
    return float_vec(m_upper.x(), m_upper.y());
  }
  constexpr float_vec lower_left() const
  {
    return float_vec(m_lower.x(), m_lower.y());
  }
  constexpr float_vec lower_right() const
  {
    return float_vec(m_upper.x(), m_lower.y());
  }
};

class int_box {
  int_vec m_begin;
  int_vec m_end;
 public:
  constexpr int_box() = default;
  constexpr int_box(
      int_vec const& begin_arg,
      int_vec const& end_arg)
    :m_begin(begin_arg)
    ,m_end(end_arg)
  {
  }
  constexpr int_box(float_box const& fb)
    :m_begin(
        int(std::floor(fb.lower().x())),
        int(std::floor(fb.lower().y())))
    ,m_end(
        int(std::ceil(fb.upper().x())),
        int(std::ceil(fb.upper().y())))
  {
  }
  constexpr int_vec const& begin() const { return m_begin; }
  constexpr int_vec const& end() const { return m_end; }
};

inline constexpr int_box
intersect(int_box const& a, int_box const& b)
{
  return int_box(
      int_vec(
        max(a.begin().x(), b.begin().x()),
        max(a.begin().y(), b.begin().y())),
      int_vec(
        min(a.end().x(), b.end().x()),
        min(a.end().y(), b.end().y())));
}

inline constexpr float default_text_width = 10.0f;
inline constexpr float default_thickness = 2.0f;

enum class text_anchor {
  top,
  bottom,
  left,
  right
};

template <class Functor>
inline void for_each(int_box const& box, Functor const& f)
{
  int_vec p;
  for (p.y() = box.end().y() - 1; p.y() >= box.begin().y(); --(p.y())) {
    for (p.x() = box.begin().x(); p.x() < box.end().x(); ++(p.x())) {
      f(p);
    }
  }
}

class image {
  unsigned m_height;
  unsigned m_width;
  std::vector<unsigned char> m_contents;
 public:
  image(unsigned width, unsigned height);
 private:
  inline unsigned char* pixel_contents(int_vec const& pixel)
  {
    return m_contents.data() + ((m_height - pixel.y() - 1) * m_width + pixel.x()) * 4;
  }
  inline unsigned char const* pixel_contents(int_vec const& pixel) const
  {
    return m_contents.data() + ((m_height - pixel.y() - 1) * m_width + pixel.x()) * 4;
  }
 public:
  inline void set(int_vec const& pixel, byte_rgba const& value)
  {
    value.store(pixel_contents(pixel));
  }
  inline byte_rgba get(int_vec const& pixel) const
  {
    return byte_rgba::load(pixel_contents(pixel));
  }
  void write(std::filesystem::path const& path) const;
  int_vec begin_pixel() const;
  int_vec end_pixel() const;
  int_box pixel_box() const;
  unsigned width() const { return m_width; }
  unsigned height() const { return m_height; }
};

image::image(unsigned width, unsigned height)
{
  m_width = width;
  m_height = height;
  m_contents.resize(width * height * 4);
}

void image::write(std::filesystem::path const& path) const
{
  unsigned result = lodepng::encode(path.string(), m_contents, m_width, m_height);
  if (result != 0) {
    throw std::runtime_error(lodepng_error_text(result));
  }
}

int_vec image::begin_pixel() const
{
  return int_vec(0, 0);
}

int_vec image::end_pixel() const
{
  return int_vec(m_width, m_height);
}

int_box image::pixel_box() const
{
  return int_box(begin_pixel(), end_pixel());
}

/*
  Curve Concept:

  class Curve {
   public:
    // the bounds of the parametric space
    constexpr float minimum_t() const;
    constexpr float maximum_t() const;
    // how many sub-segments to divide the full parametric space
    // into when using Newton's method to search for closest points
    constexpr int search_segments() const;
    // given the parameter (t), compute the following:
    // x: the point on the curve at parameter value (t)
    // dx_dt: the derivative of (x) with respect to (t)
    // d2x_dt2: the second derivative of (x) with respect to (t)
    constexpr void evaluate(
        float t,
        float_vec& x,
        float_vec& dx_dt,
        float_vec& d2x_dt2) const;
    constexpr float_box bounding_box() const;
  };
*/

class line {
  float_vec m_start;
  float_vec m_end;
 public:
  constexpr line(float_vec const& start_arg, float_vec const& end_arg)
    :m_start(start_arg)
    ,m_end(end_arg)
  {
  }
  constexpr float minimum_t() const { return 0; }
  constexpr float maximum_t() const { return 1; }
  constexpr int seed_points() const { return 2; }
  constexpr void evaluate(
      float t,
      float_vec& x,
      float_vec& dx_dt,
      float_vec& d2x_dt2) const
  {
    d2x_dt2 = float_vec(0, 0);
    dx_dt = (m_end - m_start);
    x = (1.0f - t) * m_start + t * m_end;
  }
  constexpr float_box bounding_box() const
  {
    float_box bb;
    bb.include(m_start);
    bb.include(m_end);
    return bb;
  }
};

class arc {
  float_vec m_start;
  float_vec m_center;
  float_vec m_end;
 public:
  constexpr arc(
      float_vec const& start_arg,
      float_vec const& center_arg,
      float_vec const& end_arg)
    :m_start(start_arg)
    ,m_center(center_arg)
    ,m_end(end_arg)
  {
  }
  constexpr float minimum_t() const { return 0; }
  constexpr float maximum_t() const {
    float_vec const start_vec = m_start - m_center;
    float_vec const end_vec = m_end - m_center;
    float const start_angle = std::atan2(start_vec.y(), start_vec.x());
    float end_angle = std::atan2(end_vec.y(), end_vec.x());
    if (end_angle < start_angle) end_angle += 2 * pi;
    return end_angle - start_angle;
  }
  constexpr int seed_points() const
  {
    return int(std::ceil(maximum_t() / (pi / 2)) + 1);
  }
  constexpr void evaluate(
      float t,
      float_vec& x,
      float_vec& dx_dt,
      float_vec& d2x_dt2) const
  {
    float_vec const start_vec = m_start - m_center;
    float const ct = std::cos(t);
    float const st = std::sin(t);
    float_vec const r(
        start_vec.x() * ct - start_vec.y() * st,
        start_vec.x() * st + start_vec.y() * ct);
    x = m_center + r;
    dx_dt = float_vec(
      - start_vec.x() * st - start_vec.y() * ct,
        start_vec.x() * ct - start_vec.y() * st);
    d2x_dt2 = float_vec(
      - start_vec.x() * ct + start_vec.y() * st,
      - start_vec.x() * st - start_vec.y() * ct);
  }
};

// https://en.wikipedia.org/wiki/B%C3%A9zier_curve
// https://fontforge.org/docs/techref/bezier.html

class cubic_bezier;

class quadratic_bezier {
  float_vec m_p0;
  float_vec m_p1;
  float_vec m_p2;
 public:
  constexpr quadratic_bezier(
      float_vec const& p0_arg,
      float_vec const& p1_arg,
      float_vec const& p2_arg)
    :m_p0(p0_arg)
    ,m_p1(p1_arg)
    ,m_p2(p2_arg)
  {
  }
  constexpr quadratic_bezier(cubic_bezier const&);
  constexpr float_vec const& p0() const { return m_p0; }
  constexpr float_vec const& p1() const { return m_p1; }
  constexpr float_vec const& p2() const { return m_p2; }
  constexpr float minimum_t() const { return 0.0f; }
  constexpr float maximum_t() const { return 1.0f; }
  constexpr int seed_points() const { return 3; }
  constexpr void evaluate(
      float t,
      float_vec& x,
      float_vec& dx_dt,
      float_vec& d2x_dt2) const
  {
    float const u = (1.0f - t);
    float_vec const p10 = m_p0 - m_p1;
    float_vec const p12 = m_p2 - m_p1;
    x = m_p1 + square(u) * p10 + square(t) * p12;
    dx_dt = (-2.0f * u) * p10 + (2.0f * t) * p12;
    d2x_dt2 = 2.0f * (p10 + p12);
  }
  constexpr float_box bounding_box() const
  {
    float_box bb;
    bb.include(m_p0);
    bb.include(m_p1);
    bb.include(m_p2);
    return bb;
  }
};

class cubic_bezier {
  float_vec m_p0;
  float_vec m_p1;
  float_vec m_p2;
  float_vec m_p3;
 public:
  constexpr cubic_bezier(
      float_vec const& p0_arg,
      float_vec const& p1_arg,
      float_vec const& p2_arg,
      float_vec const& p3_arg)
    :m_p0(p0_arg)
    ,m_p1(p1_arg)
    ,m_p2(p2_arg)
    ,m_p3(p3_arg)
  {
  }
  constexpr float_vec const& p0() const { return m_p0; }
  constexpr float_vec const& p1() const { return m_p1; }
  constexpr float_vec const& p2() const { return m_p2; }
  constexpr float_vec const& p3() const { return m_p3; }
  constexpr float minimum_t() const { return 0.0f; }
  constexpr float maximum_t() const { return 1.0f; }
  constexpr int seed_points() const { return 4; }
  constexpr void evaluate(
      float t,
      float_vec& x,
      float_vec& dx_dt,
      float_vec& d2x_dt2) const
  {
    x = cube(1.0f - t) * m_p0
      + (3.0f * square(1.0f - t) * t) * m_p1
      + (3.0f * (1.0f - t) * square(t)) * m_p2
      + cube(t) * m_p3;
    dx_dt = (3.0f * square(1.0f - t)) * (m_p1 - m_p0)
          + (6.0f * (1.0f - t) * t) * (m_p2 - m_p1)
          + (3.0f * square(t)) * (m_p3 - m_p2);
    d2x_dt2 = (6.0f * (1.0f - t)) * (m_p2 - 2.0f * m_p1 + m_p0)
            + (6.0f * t) * (m_p3 - 2.0f * m_p2 + m_p1);
  }
};

// CP0 = QP0
// CP1 = QP0 + (2/3) * (QP1 - QP0)
// CP2 = QP2 + (2/3) * (QP1 - QP2)
// CP3 = QP2

// QP1 = CP0 + (3/2) * (CP1 - CP0)
// QP1 = QP0 + (3/2) * (QP0 + (2/3) * (QP1 - QP0) - QP0)
// QP1 = QP0 + (3/2) * QP0 + (QP1 - QP0) - (3/2) * QP0

constexpr quadratic_bezier::quadratic_bezier(cubic_bezier const& c)
  :m_p0(c.p0())
  ,m_p1(c.p0() + (3.0f / 2.0f) * (c.p1() - c.p0()))
  ,m_p2(c.p3())
{
}

// squared_distance = (curve_x - query_x) . (curve_x - query_x)
// squared_distance(t) = (curve_x(t) - query_x) . (curve_x(t) - query_x)
// squared_distance(t) = (x(t) - qx) * (x(t) - qx) + (y(t) - qy) * (y(t) - qy)
// squared_distance(t) = x(t)^2 - 2 * x(t) * qx + qx^2
//                     + y(t)^2 - 2 * y(t) * qy + qy^2
// squared_distance_deriv(t) = 2 * x(t) * dx_dt(t) - 2 * dx_dt(t) * qx
//                           + 2 * y(t) * dy_dt(t) - 2 * dy_dt(t) * qy
// squared_distance_deriv(t) = 2 * (curve_x(t) - query_x) . curve_deriv(t)
// squared_distance_second_deriv(t) =
//    2 * dx_dt(t) * dx_dt(t) + 2 * x(t) * d2x_dt2(t) - 2 * d2x_dt2(t) * qx
//  + 2 * dy_dt(t) * dy_dt(t) + 2 * y(t) * d2y_dt2(t) - 2 * d2y_dt2(t) * qy
// squared_distance_second_deriv(t) =
//    2 * dx_dt(t) . dx_dt(t) + 2 * (x(t) - qx) . d2x_dt2(t)

// given a point on a curve and the first and second derivatives of the curve,
// compute the squared distance from the point to a query point (q),
// as well as the first and second derivatives of the squared distance function

inline constexpr void evaluate_distance_squared(
    float_vec const& x,
    float_vec const& dx_dt,
    float_vec const& d2x_dt2,
    float_vec const& q,
    float& l2,
    float& dl2_dt,
    float& d2l2_dt2)
{
  float_vec const xmq = x - q;
  l2 = squared_magnitude(xmq);
  dl2_dt = 2.0f * dot(xmq, dx_dt);
  d2l2_dt2 = 2.0f * dot(dx_dt, dx_dt) + 2.0f * dot(xmq, d2x_dt2);
}

// given a curve object and parametric coordinate (t),
// compute the squared distance from the point to a query point (q),
// as well as the first and second derivatives of the squared distance function

template <class Curve>
inline constexpr void evaluate_distance_squared(
    Curve const& curve,
    float_vec const& q,
    float t,
    float_vec& x,
    float_vec& dx_dt,
    float& l2,
    float& dl2_dt,
    float& d2l2_dt2)
{
  float_vec d2x_dt2;
  curve.evaluate(t, x, dx_dt, d2x_dt2);
  evaluate_distance_squared(x, dx_dt, d2x_dt2, q, l2, dl2_dt, d2l2_dt2);
}

// CAD-kernel-like absolute tolerance.
// points closer than this are considered "the same".
// this is in pixel units, so one thousandth of a pixel.
inline constexpr float absolute_tolerance = 1.0e-3;

// given a curve object and a parametric range [min_t, max_t],
// compute a value of (t) in the range that evaluates to
// a point which is closest to a query point (q).
// also returns the distance.
// this function will stop when it is closer to the query point
// than the absolute tolerance, or when its Newton search moves
// closer by less than the absolute tolerance.

template <class Curve>
inline void find_closest_by_newton(
    Curve const& curve,
    float_vec const& q,
    float& t,
    float& l2,
    float_vec& point,
    float_vec& tangent)
{
  int constexpr max_iterations = 10;
  float constexpr squared_tolerance = square(absolute_tolerance);
  float old_l2 = std::numeric_limits<float>::max();
  float old_t = std::numeric_limits<float>::quiet_NaN();
  float_vec old_point;
  float_vec old_tangent;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    float dl2_dt;
    float d2l2_dt2;
    evaluate_distance_squared(curve, q, t, point, tangent, l2, dl2_dt, d2l2_dt2);
    if (l2 > old_l2) {
      // starting to diverge, escape.
      t = old_t;
      l2 = old_l2;
      point = old_point;
      tangent = old_tangent;
      return;
    }
    if (l2 <= squared_tolerance) {
      // per the tolerance, we are at the query point itself.
      // no need to look further.
      return;
    }
    if (iteration > 1) {
      float const squared_distance_moved =
        squared_magnitude(point - old_point);
      if (squared_distance_moved <= squared_tolerance) {
        // we haven't moved very much, we're probably converged
        return;
      }
    }
    if (d2l2_dt2 == 0.0f) {
      return;
    }
    // https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
    float const newton_t = t - dl2_dt / d2l2_dt2;
    old_t = t;
    old_l2 = l2;
    old_point = point;
    old_tangent = tangent;
    t = clamp(newton_t, 0.0f, 1.0f);
  }
}

// uses the above Newton search on sub-segments of the
// parametric range of the curve, and returns the minimum
// of the results from each sub-segment.

template <class Curve>
inline void find_closest(
    Curve const& curve,
    float_vec const& q,
    float& t,
    float& l2,
    float_vec& point,
    float_vec& tangent)
{
  int const seed_points = curve.seed_points();
  l2 = std::numeric_limits<float>::max();
  t = std::numeric_limits<float>::quiet_NaN();
  for (int i = 0; i < seed_points; ++i) {
    float newton_t = float(i) / float(seed_points - 1);
    float newton_l2;
    float_vec newton_point, newton_tangent;
    find_closest_by_newton(curve, q,
        newton_t,
        newton_l2,
        newton_point,
        newton_tangent);
    if (newton_l2 < l2) {
      l2 = newton_l2;
      t = newton_t;
      point = newton_point;
      tangent = newton_tangent;
    }
  }
}

inline constexpr bool is_inside(
    float_vec const& tangent,
    float_vec const& trial_vector)
{
  return dot(trial_vector, rotate_right_90deg(tangent)) > 0.0f;
}

enum class intersection {
  fully_inside,
  fully_outside,
  partially_inside
};

class outline {
  std::vector<line> m_lines;
  std::vector<quadratic_bezier> m_quadratic_beziers;
 public:
  void clear()
  {
    m_lines.clear();
    m_quadratic_beziers.clear();
  }
  void add(line const& l)
  {
    m_lines.push_back(l);
  }
  void add(quadratic_bezier const& q)
  {
    m_quadratic_beziers.push_back(q);
  }
 private:
  template <class T>
  void find_closest_single_type(
      float_vec const& q,
      std::vector<T> const& curves,
      float& closest_l2,
      float_vec& closest_point,
      float_vec& closest_tangent) const
  {
    for (std::size_t i = 0; i < curves.size(); ++i) {
      float trial_t, trial_l2;
      float_vec trial_point, trial_tangent;
      find_closest(curves[i], q, trial_t, trial_l2, trial_point, trial_tangent);
      if (trial_l2 > closest_l2) {
        // this curve is further from the query point than
        // other curves already found, skip it.
        continue;
      }
      if (trial_l2 < closest_l2) {
        // the current curve is closer than anything else found so far,
        // just overwrite the "closest" variables
        closest_l2 = trial_l2;
        closest_point = trial_point;
        closest_tangent = trial_tangent;
      } else {
        // by elimination, in this code block we are dealing with
        // a curve that has the same distance from the query point
        // as the previous closest curve.
        if (closest_point == trial_point) {
          // in this code block, the closest point on this curve is
          // furthermore the same point as the previous closest point.
          // this should only occur when the closest point is a shared corner
          // that is the endpoint of two curves.
          // it now becomes fairly tricky to determine "insideness" based on
          // the tangent of only one curve.
          // we resolve this by noting that the closest point on the curves
          // can only be the shared endpoint when the query point is in the
          // "large angle" of the corner (the angle greater than 180 degrees).
          // therefore, whether the query point is inside the shape or not
          // amounts to figuring out whether the "large angle" is inside
          // the shape or not, which is the opposite of whether the "small angle"
          // is inside.
          // this depends entirely on the direction of the curves themselves.
          // we'll replace the "closest_tangent" with a vector that will indicate
          // inside or outside based on what we find.
          float_vec const closest_to_query = q - closest_point;
          if (is_inside(trial_tangent, closest_to_query)
              != is_inside(closest_tangent, closest_to_query)) {
            float_vec new_tangent = rotate_left_90deg(closest_to_query);
            float_vec first_tangent;
            float_vec second_tangent;
            if (trial_t == 1.0f) {
              first_tangent = trial_tangent;
              second_tangent = closest_tangent;
            } else {
              first_tangent = closest_tangent;
              second_tangent = trial_tangent;
            }
            if (is_inside(first_tangent, second_tangent)) {
              // the "small angle" is inside the outline so the "large angle"
              // must be outside, negate our new fake tangent
              new_tangent = -new_tangent;
            }
            closest_tangent = new_tangent;
          }
        }
      }
    }
  }
  template <class T>
  float_box bounding_box_single_type(std::vector<T> const& curves) const
  {
    float_box result;
    for (T const& curve : curves) {
      result.include(curve.bounding_box());
    }
    return result;
  }
 public:
  float_box bounding_box() const
  {
    float_box result;
    result.include(bounding_box_single_type(m_lines));
    result.include(bounding_box_single_type(m_quadratic_beziers));
    return result;
  }
  int_box bounding_pixels() const
  {
    return int_box(bounding_box());
  }
  intersection intersect(circle const& c) const
  {
    float l2 = std::numeric_limits<float>::max();
    float_vec closest_point, tangent;
    find_closest_single_type(c.center(), m_lines, l2, closest_point, tangent);
    find_closest_single_type(c.center(), m_quadratic_beziers, l2, closest_point, tangent);
    if (l2 < c.radius_squared()) {
      return intersection::partially_inside;
    }
    if (is_inside(tangent, c.center() - closest_point)) {
      return intersection::fully_inside;
    } else {
      return intersection::fully_outside;
    }
  }
};

inline float recursive_sample_float_box(
    coloring::outline const& outline,
    float_box const& box,
    int remaining_depth)
{
  auto const circle = box.bounding_circle();
  auto const status = outline.intersect(circle);
  auto const l = box.lower();
  auto const m = circle.center();
  auto const h = box.upper();
  if (status == intersection::fully_inside) return 1.0f;
  if (status == intersection::fully_outside) return 0.0f;
  if (remaining_depth <= 0) return 0.5f;
  float value = 0.0f;
  value += recursive_sample_float_box(
      outline,
      float_box({l.x(), l.y()}, {m.x(), m.y()}),
      remaining_depth - 1);
  value += recursive_sample_float_box(
      outline,
      float_box({m.x(), l.y()}, {h.x(), m.y()}),
      remaining_depth - 1);
  value += recursive_sample_float_box(
      outline,
      float_box({l.x(), m.y()}, {m.x(), h.y()}),
      remaining_depth - 1);
  value += recursive_sample_float_box(
      outline,
      float_box({m.x(), m.y()}, {h.x(), h.y()}),
      remaining_depth - 1);
  value *= 0.25f;
  return value;
}

inline constexpr int subpixel_depth = 3;

inline float sample_pixel(coloring::outline const& outline, int_vec const& pixel)
{
  auto const box = float_box(
      float_vec(pixel.x(), pixel.y()),
      float_vec(pixel.x() + 1, pixel.y() + 1));
  auto const result = recursive_sample_float_box(outline, box, subpixel_depth);
  return result;
}

inline void draw(
    coloring::image& image,
    outline const& shape,
    byte_rgba const& color)
{
  auto const pixel_box = intersect(
      shape.bounding_pixels(),
      image.pixel_box());
  for_each(pixel_box,
  [&] (int_vec const& p) {
    float const fraction = sample_pixel(shape, p);
    auto pixel_rgba = image.get(p);
    pixel_rgba = byte_rgba(color.rgb(), btof(color.a()) * fraction).over(pixel_rgba);
    image.set(p, pixel_rgba);
  });
}

class font {
 public:
  static constexpr float original_width{1229};
  static constexpr float lowest_y{-425};
  static constexpr float highest_y{1484};
  static constexpr float original_height{highest_y - lowest_y};
  static constexpr float aspect_ratio{original_height / original_width};
 private:
  float_vec m_current_position;
  outline* m_outline_ptr{nullptr};
  float m_scale{0.0f};
  float m_width{0.0f};
  float m_height{0.0f};
  float_vec m_origin{0.0f, 0.0f};
  bool m_vertical{false};
 public:
  font(
      outline& outline_arg,
      float_vec const& origin_arg,
      float width_arg,
      bool vertical_arg)
    :m_outline_ptr(&outline_arg)
    ,m_scale(width_arg / original_width)
    ,m_width(width_arg)
    ,m_height(m_scale * original_height)
    ,m_origin(origin_arg)
    ,m_vertical(vertical_arg)
  {
  }
  float height() const { return m_height; }
  void advance()
  {
    float_vec v(m_width, 0);
    if (m_vertical) v = rotate_left_90deg(v);
    m_origin = m_origin + v;
  }
  float_vec transform(float x, float y) const
  {
    float_vec v = m_scale * float_vec(x, y - lowest_y); 
    if (m_vertical) v = rotate_left_90deg(v);
    return m_origin + v;
  }
  void moveto(float x, float y)
  {
    m_current_position = transform(x, y);
  }
  void lineto(float x, float y)
  {
    auto const new_position = transform(x, y);
    m_outline_ptr->add(line(m_current_position, new_position));
    m_current_position = new_position;
  }
  void curveto(
      float x1, float y1,
      float x2, float y2)
  {
    auto const new_position = transform(x2, y2);
    auto const q = quadratic_bezier(
        m_current_position,
        transform(x1, y1),
        new_position);
    m_outline_ptr->add(q);
    m_current_position = new_position;
  }
// begin hardcoded LiberationMono-Regular font!
  void add_exclam() {
    moveto(689, 397);
    lineto(541, 397);
    lineto(517, 1348);
    lineto(713, 1348);
    lineto(689, 397);
    moveto(515, 0);
    lineto(515, 201);
    lineto(709, 201);
    lineto(709, 0);
    lineto(515, 0);
  }
  void add_quotedbl() {
    moveto(908, 845);
    lineto(766, 845);
    lineto(726, 1484);
    lineto(950, 1484);
    lineto(908, 845);
    moveto(459, 845);
    lineto(318, 845);
    lineto(277, 1484);
    lineto(501, 1484);
    lineto(459, 845);
  }
  void add_numbersign() {
    moveto(930, 833);
    lineto(863, 516);
    lineto(1123, 516);
    lineto(1123, 408);
    lineto(840, 408);
    lineto(752, 0);
    lineto(642, 0);
    lineto(728, 408);
    lineto(365, 408);
    lineto(281, 0);
    lineto(171, 0);
    lineto(255, 408);
    lineto(54, 408);
    lineto(54, 516);
    lineto(279, 516);
    lineto(346, 833);
    lineto(105, 833);
    lineto(105, 941);
    lineto(368, 941);
    lineto(457, 1349);
    lineto(567, 1349);
    lineto(479, 941);
    lineto(842, 941);
    lineto(930, 1349);
    lineto(1040, 1349);
    lineto(952, 941);
    lineto(1163, 941);
    lineto(1163, 833);
    lineto(930, 833);
    moveto(459, 833);
    lineto(390, 516);
    lineto(752, 516);
    lineto(819, 833);
    lineto(459, 833);
  }
  void add_dollar() {
    moveto(1150, 380);
    curveto(1150, 218, 1030, 123.5);
    curveto(910, 29, 686, 20);
    lineto(686, -141);
    lineto(558, -141);
    lineto(558, 20);
    curveto(136, 34, 66, 379);
    lineto(236, 416);
    curveto(262, 290, 342, 227.5);
    curveto(422, 165, 558, 158);
    lineto(558, 647);
    lineto(528, 655);
    curveto(360, 696, 281.5, 744.5);
    curveto(203, 793, 166.5, 861);
    curveto(130, 929, 130, 1023);
    curveto(130, 1172, 239.5, 1255.5);
    curveto(349, 1339, 558, 1346);
    lineto(558, 1476);
    lineto(686, 1476);
    lineto(686, 1346);
    curveto(872, 1339, 973, 1266.5);
    curveto(1074, 1194, 1119, 1025);
    lineto(945, 992);
    curveto(923, 1098, 859, 1152);
    curveto(795, 1206, 686, 1213);
    lineto(686, 787);
    curveto(858, 745, 935, 709);
    curveto(1012, 673, 1058, 627);
    curveto(1104, 581, 1127, 520.5);
    curveto(1150, 460, 1150, 380);
    moveto(302, 1018);
    curveto(302, 961, 325.5, 922.5);
    curveto(349, 884, 394, 858.5);
    curveto(439, 833, 558, 802);
    lineto(558, 1215);
    curveto(432, 1210, 367, 1159.5);
    curveto(302, 1109, 302, 1018);
    moveto(978, 383);
    curveto(978, 449, 950.5, 492);
    curveto(923, 535, 873, 563.5);
    curveto(823, 592, 686, 627);
    lineto(686, 156);
    curveto(830, 165, 904, 223.5);
    curveto(978, 282, 978, 383);
  }
  void add_percent() {
    moveto(221, 0);
    lineto(76, 0);
    lineto(1008, 1353);
    lineto(1155, 1353);
    lineto(221, 0);
    moveto(291, 1361);
    curveto(574, 1361, 574, 1025);
    curveto(574, 858, 501, 771);
    curveto(428, 684, 287, 684);
    curveto(146, 684, 73, 770);
    curveto(0, 856, 0, 1025);
    curveto(0, 1192, 70, 1276.5);
    curveto(140, 1361, 291, 1361);
    moveto(427, 1025);
    curveto(427, 1144, 395.5, 1198.5);
    curveto(364, 1253, 290, 1253);
    curveto(213, 1253, 180, 1200);
    curveto(147, 1147, 147, 1025);
    curveto(147, 909, 180, 853);
    curveto(213, 797, 289, 797);
    curveto(360, 797, 393.5, 852.5);
    curveto(427, 908, 427, 1025);
    moveto(947, 665);
    curveto(1230, 665, 1230, 329);
    curveto(1230, 162, 1157, 75);
    curveto(1084, -12, 943, -12);
    curveto(802, -12, 729, 74);
    curveto(656, 160, 656, 329);
    curveto(656, 496, 726, 580.5);
    curveto(796, 665, 947, 665);
    moveto(1083, 329);
    curveto(1083, 448, 1051.5, 502.5);
    curveto(1020, 557, 946, 557);
    curveto(869, 557, 836, 504);
    curveto(803, 451, 803, 329);
    curveto(803, 213, 836, 157);
    curveto(869, 101, 945, 101);
    curveto(1016, 101, 1049.5, 156.5);
    curveto(1083, 212, 1083, 329);
  }
  void add_ampersand() {
    moveto(1072, -12);
    curveto(999, -12, 928.5, 23);
    curveto(858, 58, 808, 117);
    curveto(655, -20, 461, -20);
    curveto(262, -20, 152.5, 80);
    curveto(43, 180, 43, 358);
    curveto(43, 626, 335, 777);
    curveto(303, 837, 281.5, 913.5);
    curveto(260, 990, 260, 1059);
    curveto(260, 1195, 352.5, 1276);
    curveto(445, 1357, 611, 1357);
    curveto(757, 1357, 844.5, 1285);
    curveto(932, 1213, 932, 1090);
    curveto(932, 1015, 892.5, 956.5);
    curveto(853, 898, 778.5, 847.5);
    curveto(704, 797, 525, 722);
    curveto(657, 491, 802, 333);
    curveto(890, 515, 927, 739);
    lineto(1072, 696);
    curveto(1028, 460, 910, 234);
    curveto(953, 180, 1001.5, 154.5);
    curveto(1050, 129, 1090, 129);
    curveto(1148, 129, 1185, 145);
    lineto(1185, 10);
    curveto(1137, -12, 1072, -12);
    moveto(780, 1085);
    curveto(780, 1155, 732, 1195.5);
    curveto(684, 1236, 608, 1236);
    curveto(519, 1236, 463.5, 1187);
    curveto(408, 1138, 408, 1056);
    curveto(408, 946, 469, 832);
    curveto(630, 900, 679, 931.5);
    curveto(728, 963, 754, 1000.5);
    curveto(780, 1038, 780, 1085);
    moveto(706, 217);
    curveto(526, 420, 390, 658);
    curveto(211, 561, 211, 362);
    curveto(211, 247, 280, 179);
    curveto(349, 111, 469, 111);
    curveto(533, 111, 595.5, 138.5);
    curveto(658, 166, 706, 217);
  }
  void add_quotesingle() {
    moveto(684, 845);
    lineto(543, 845);
    lineto(502, 1484);
    lineto(726, 1484);
    lineto(684, 845);
  }
  void add_parenleft() {
    moveto(529, 530);
    curveto(529, 255, 614, 33);
    curveto(699, -189, 891, -425);
    lineto(701, -425);
    curveto(509, -189, 425.5, 32.5);
    curveto(342, 254, 342, 532);
    curveto(342, 805, 424, 1024.5);
    curveto(506, 1244, 701, 1484);
    lineto(891, 1484);
    curveto(699, 1248, 614, 1025.5);
    curveto(529, 803, 529, 530);
  }
  void add_parenright() {
    moveto(885, 532);
    curveto(885, 252, 802.5, 31.5);
    curveto(720, -189, 528, -425);
    lineto(336, -425);
    curveto(532, -184, 616, 38.5);
    curveto(700, 261, 700, 530);
    curveto(700, 798, 616, 1020.5);
    curveto(532, 1243, 336, 1484);
    lineto(528, 1484);
    curveto(723, 1244, 804, 1026);
    curveto(885, 808, 885, 532);
  }
  void add_asterisk() {
    moveto(671, 1188);
    lineto(935, 1291);
    lineto(980, 1159);
    lineto(698, 1086);
    lineto(883, 836);
    lineto(764, 764);
    lineto(614, 1022);
    lineto(458, 766);
    lineto(339, 838);
    lineto(528, 1086);
    lineto(248, 1159);
    lineto(293, 1293);
    lineto(560, 1186);
    lineto(548, 1483);
    lineto(684, 1483);
    lineto(671, 1188);
  }
  void add_plus() {
    moveto(687, 608);
    lineto(687, 180);
    lineto(540, 180);
    lineto(540, 608);
    lineto(116, 608);
    lineto(116, 754);
    lineto(540, 754);
    lineto(540, 1182);
    lineto(687, 1182);
    lineto(687, 754);
    lineto(1111, 754);
    lineto(1111, 608);
    lineto(687, 608);
  }
  void add_comma() {
    moveto(259, -363);
    lineto(428, 299);
    lineto(693, 299);
    lineto(382, -363);
    lineto(259, -363);
  }
  void add_hyphen() {
    moveto(334, 464);
    lineto(334, 624);
    lineto(894, 624);
    lineto(894, 464);
    lineto(334, 464);
  }
  void add_period() {
    moveto(496, 0);
    lineto(496, 299);
    lineto(731, 299);
    lineto(731, 0);
    lineto(496, 0);
  }
  void add_slash() {
    moveto(114, -20);
    lineto(935, 1484);
    lineto(1113, 1484);
    lineto(296, -20);
    lineto(114, -20);
  }
  void add_zero() {
    moveto(1103, 675);
    curveto(1103, 337, 978.5, 158.5);
    curveto(854, -20, 611, -20);
    curveto(368, -20, 246, 157.5);
    curveto(124, 335, 124, 675);
    curveto(124, 1024, 243, 1197);
    curveto(362, 1370, 617, 1370);
    curveto(866, 1370, 984.5, 1195.5);
    curveto(1103, 1021, 1103, 675);
    moveto(920, 675);
    curveto(920, 965, 849.5, 1094.5);
    curveto(779, 1224, 617, 1224);
    curveto(451, 1224, 378.5, 1096);
    curveto(306, 968, 306, 675);
    curveto(306, 390, 379.5, 258.5);
    curveto(453, 127, 613, 127);
    curveto(772, 127, 846, 262);
    curveto(920, 397, 920, 675);
    moveto(496, 555);
    lineto(496, 804);
    lineto(731, 804);
    lineto(731, 555);
    lineto(496, 555);
  }
  void add_one() {
    moveto(157, 0);
    lineto(157, 145);
    lineto(596, 145);
    lineto(596, 1166);
    curveto(559, 1088, 420.5, 1030);
    curveto(282, 972, 148, 972);
    lineto(148, 1120);
    curveto(296, 1120, 427.5, 1185);
    curveto(559, 1250, 611, 1349);
    lineto(777, 1349);
    lineto(777, 145);
    lineto(1130, 145);
    lineto(1130, 0);
    lineto(157, 0);
  }
  void add_two() {
    moveto(144, 0);
    lineto(144, 117);
    curveto(193, 226, 296.5, 336.5);
    curveto(400, 447, 578, 589);
    curveto(737, 716, 807, 810);
    curveto(877, 904, 877, 991);
    curveto(877, 1102, 808, 1162);
    curveto(739, 1222, 611, 1222);
    curveto(497, 1222, 426.5, 1159.5);
    curveto(356, 1097, 343, 984);
    lineto(159, 1001);
    curveto(179, 1171, 298, 1270.5);
    curveto(417, 1370, 611, 1370);
    curveto(824, 1370, 943, 1274);
    curveto(1062, 1178, 1062, 1002);
    curveto(1062, 887, 986, 772.5);
    curveto(910, 658, 759, 538);
    curveto(553, 374, 473.5, 296.5);
    curveto(394, 219, 361, 146);
    lineto(1084, 146);
    lineto(1084, 0);
    lineto(144, 0);
  }
  void add_three() {
    moveto(1099, 370);
    curveto(1099, 184, 973, 82);
    curveto(847, -20, 621, -20);
    curveto(407, -20, 279, 77);
    curveto(151, 174, 128, 362);
    lineto(314, 379);
    curveto(350, 129, 621, 129);
    curveto(757, 129, 834.5, 192);
    curveto(912, 255, 912, 376);
    curveto(912, 451, 866.5, 502.5);
    curveto(821, 554, 743, 581.5);
    curveto(665, 609, 568, 609);
    lineto(466, 609);
    lineto(466, 765);
    lineto(564, 765);
    curveto(650, 765, 721.5, 793.5);
    curveto(793, 822, 834, 874);
    curveto(875, 926, 875, 997);
    curveto(875, 1103, 808.5, 1162.5);
    curveto(742, 1222, 611, 1222);
    curveto(492, 1222, 418.5, 1161);
    curveto(345, 1100, 333, 989);
    lineto(152, 1003);
    curveto(172, 1176, 295.5, 1273);
    curveto(419, 1370, 613, 1370);
    curveto(825, 1370, 942.5, 1276.5);
    curveto(1060, 1183, 1060, 1016);
    curveto(1060, 897, 981, 809);
    curveto(902, 721, 765, 693);
    lineto(765, 689);
    curveto(916, 672, 1007.5, 583);
    curveto(1099, 494, 1099, 370);
  }
  void add_four() {
    moveto(937, 319);
    lineto(937, 0);
    lineto(757, 0);
    lineto(757, 319);
    lineto(103, 319);
    lineto(103, 459);
    lineto(738, 1349);
    lineto(937, 1349);
    lineto(937, 461);
    lineto(1125, 461);
    lineto(1125, 319);
    lineto(937, 319);
    moveto(757, 1154);
    lineto(257, 461);
    lineto(757, 461);
    lineto(757, 1154);
  }
  void add_five() {
    moveto(1099, 444);
    curveto(1099, 305, 1040, 200);
    curveto(981, 95, 867.5, 37.5);
    curveto(754, -20, 599, -20);
    curveto(402, -20, 281, 66);
    curveto(160, 152, 128, 315);
    lineto(310, 336);
    curveto(367, 127, 603, 127);
    curveto(744, 127, 828, 211);
    curveto(912, 295, 912, 440);
    curveto(912, 564, 829, 643);
    curveto(746, 722, 607, 722);
    curveto(534, 722, 471, 699);
    curveto(408, 676, 345, 621);
    lineto(169, 621);
    lineto(216, 1349);
    lineto(1017, 1349);
    lineto(1017, 1204);
    lineto(382, 1204);
    lineto(353, 779);
    curveto(470, 869, 644, 869);
    curveto(848, 869, 973.5, 751.5);
    curveto(1099, 634, 1099, 444);
  }
  void add_six() {
    moveto(1096, 446);
    curveto(1096, 234, 974.5, 107);
    curveto(853, -20, 641, -20);
    curveto(405, -20, 278, 152.5);
    curveto(151, 325, 151, 642);
    curveto(151, 990, 283, 1180);
    curveto(415, 1370, 655, 1370);
    curveto(974, 1370, 1057, 1083);
    lineto(885, 1052);
    curveto(832, 1224, 653, 1224);
    curveto(500, 1224, 415, 1085);
    curveto(330, 946, 330, 695);
    curveto(379, 786, 468, 833.5);
    curveto(557, 881, 672, 881);
    curveto(864, 881, 980, 762.5);
    curveto(1096, 644, 1096, 446);
    moveto(913, 438);
    curveto(913, 582, 836.5, 662);
    curveto(760, 742, 629, 742);
    curveto(555, 742, 489, 708.5);
    curveto(423, 675, 385.5, 615.5);
    curveto(348, 556, 348, 481);
    curveto(348, 329, 428.5, 227);
    curveto(509, 125, 635, 125);
    curveto(762, 125, 837.5, 209);
    curveto(913, 293, 913, 438);
  }
  void add_seven() {
    moveto(1069, 1210);
    curveto(596, 530, 596, 0);
    lineto(408, 0);
    curveto(408, 263, 530.5, 567.5);
    curveto(653, 872, 895, 1204);
    lineto(158, 1204);
    lineto(158, 1349);
    lineto(1069, 1349);
    lineto(1069, 1210);
  }
  void add_eight() {
    moveto(1094, 378);
    curveto(1094, 194, 969.5, 87);
    curveto(845, -20, 614, -20);
    curveto(388, -20, 260.5, 85);
    curveto(133, 190, 133, 376);
    curveto(133, 505, 212, 595.5);
    curveto(291, 686, 414, 707);
    lineto(414, 711);
    curveto(302, 738, 234, 825);
    curveto(166, 912, 166, 1024);
    curveto(166, 1122, 221.5, 1202);
    curveto(277, 1282, 378, 1326);
    curveto(479, 1370, 610, 1370);
    curveto(747, 1370, 849, 1325.5);
    curveto(951, 1281, 1005, 1202);
    curveto(1059, 1123, 1059, 1022);
    curveto(1059, 909, 990, 822);
    curveto(921, 735, 809, 713);
    lineto(809, 709);
    curveto(939, 688, 1016.5, 599.5);
    curveto(1094, 511, 1094, 378);
    moveto(872, 1012);
    curveto(872, 1123, 804.5, 1179.5);
    curveto(737, 1236, 610, 1236);
    curveto(487, 1236, 418.5, 1179);
    curveto(350, 1122, 350, 1012);
    curveto(350, 901, 419, 840);
    curveto(488, 779, 612, 779);
    curveto(872, 779, 872, 1012);
    moveto(907, 395);
    curveto(907, 515, 829, 579.5);
    curveto(751, 644, 610, 644);
    curveto(474, 644, 396.5, 574.5);
    curveto(319, 505, 319, 391);
    curveto(319, 256, 394.5, 185.5);
    curveto(470, 115, 616, 115);
    curveto(763, 115, 835, 184);
    curveto(907, 253, 907, 395);
  }
  void add_nine() {
    moveto(1087, 703);
    curveto(1087, 357, 954, 168.5);
    curveto(821, -20, 577, -20);
    curveto(412, -20, 312.5, 49.5);
    curveto(213, 119, 170, 274);
    lineto(342, 301);
    curveto(396, 125, 580, 125);
    curveto(734, 125, 820.5, 264.5);
    curveto(907, 404, 909, 650);
    curveto(869, 560, 772, 505.5);
    curveto(675, 451, 559, 451);
    curveto(370, 451, 255.5, 578.5);
    curveto(141, 706, 141, 911);
    curveto(141, 1123, 266.5, 1246.5);
    curveto(392, 1370, 610, 1370);
    curveto(1087, 1370, 1087, 703);
    moveto(891, 862);
    curveto(891, 1023, 811, 1123.5);
    curveto(731, 1224, 604, 1224);
    curveto(474, 1224, 399, 1137);
    curveto(324, 1050, 324, 911);
    curveto(324, 768, 399, 680.5);
    curveto(474, 593, 602, 593);
    curveto(678, 593, 745.5, 627.5);
    curveto(813, 662, 852, 723.5);
    curveto(891, 785, 891, 862);
  }
  void add_colon() {
    moveto(496, 0);
    lineto(496, 299);
    lineto(731, 299);
    lineto(731, 0);
    lineto(496, 0);
    moveto(496, 783);
    lineto(496, 1082);
    lineto(731, 1082);
    lineto(731, 783);
    lineto(496, 783);
  }
  void add_semicolon() {
    moveto(496, 783);
    lineto(496, 1082);
    lineto(731, 1082);
    lineto(731, 783);
    lineto(496, 783);
    moveto(352, -363);
    lineto(521, 299);
    lineto(786, 299);
    lineto(475, -363);
    lineto(352, -363);
  }
  void add_less() {
    moveto(116, 571);
    lineto(116, 776);
    lineto(1111, 1194);
    lineto(1111, 1040);
    lineto(253, 674);
    lineto(1111, 307);
    lineto(1111, 154);
    lineto(116, 571);
  }
  void add_equal() {
    moveto(116, 856);
    lineto(116, 1004);
    lineto(1111, 1004);
    lineto(1111, 856);
    lineto(116, 856);
    moveto(116, 344);
    lineto(116, 492);
    lineto(1111, 492);
    lineto(1111, 344);
    lineto(116, 344);
  }
  void add_greater() {
    moveto(116, 154);
    lineto(116, 307);
    lineto(974, 674);
    lineto(116, 1040);
    lineto(116, 1194);
    lineto(1111, 776);
    lineto(1111, 571);
    lineto(116, 154);
  }
  void add_question() {
    moveto(1073, 1002);
    curveto(1073, 929, 1051, 872);
    curveto(1029, 815, 987.5, 766.5);
    curveto(946, 718, 854, 652);
    curveto(744, 572, 708.5, 535);
    curveto(673, 498, 652.5, 455);
    curveto(632, 412, 631, 357);
    lineto(456, 357);
    curveto(458, 427, 480.5, 482.5);
    curveto(503, 538, 542.5, 584.5);
    curveto(582, 631, 677, 702);
    curveto(784, 780, 817.5, 817);
    curveto(851, 854, 871, 896.5);
    curveto(891, 939, 891, 994);
    curveto(891, 1103, 811.5, 1161.5);
    curveto(732, 1220, 596, 1220);
    curveto(461, 1220, 376.5, 1147);
    curveto(292, 1074, 278, 948);
    lineto(94, 960);
    curveto(120, 1157, 253.5, 1263.5);
    curveto(387, 1370, 594, 1370);
    curveto(817, 1370, 945, 1273.5);
    curveto(1073, 1177, 1073, 1002);
    moveto(448, 0);
    lineto(448, 201);
    lineto(643, 201);
    lineto(643, 0);
    lineto(448, 0);
  }
  void add_at() {
    moveto(1189, 755);
    curveto(1189, 457, 1106.5, 280.5);
    curveto(1024, 104, 884, 104);
    curveto(741, 104, 741, 261);
    lineto(741, 269);
    lineto(743, 299);
    lineto(737, 299);
    curveto(703, 211, 641.5, 157.5);
    curveto(580, 104, 502, 104);
    curveto(396, 104, 336.5, 204);
    curveto(277, 304, 277, 489);
    curveto(277, 650, 325.5, 793);
    curveto(374, 936, 457.5, 1018.5);
    curveto(541, 1101, 644, 1101);
    curveto(715, 1101, 760.5, 1059.5);
    curveto(806, 1018, 828, 928);
    lineto(833, 928);
    lineto(865, 1079);
    lineto(981, 1079);
    lineto(882, 572);
    curveto(842, 362, 842, 294);
    curveto(842, 206, 891, 206);
    curveto(969, 206, 1018.5, 355);
    curveto(1068, 504, 1068, 753);
    curveto(1068, 1027, 959, 1197.5);
    curveto(850, 1368, 670, 1368);
    curveto(522, 1368, 408, 1260.5);
    curveto(294, 1153, 232, 958);
    curveto(170, 763, 170, 514);
    curveto(170, 205, 286, 20);
    curveto(402, -165, 604, -165);
    curveto(698, -165, 785, -130);
    curveto(872, -95, 973, -7);
    lineto(1044, -94);
    curveto(929, -193, 821, -238);
    curveto(713, -283, 594, -283);
    curveto(429, -283, 303.5, -186);
    curveto(178, -89, 111, 93.5);
    curveto(44, 276, 44, 514);
    curveto(44, 800, 121, 1021.5);
    curveto(198, 1243, 341, 1363.5);
    curveto(484, 1484, 672, 1484);
    curveto(830, 1484, 946.5, 1393.5);
    curveto(1063, 1303, 1126, 1137.5);
    curveto(1189, 972, 1189, 755);
    moveto(775, 784);
    curveto(775, 890, 740, 944.5);
    curveto(705, 999, 646, 999);
    curveto(583, 999, 530, 932.5);
    curveto(477, 866, 444.5, 744.5);
    curveto(412, 623, 412, 491);
    curveto(412, 210, 526, 210);
    curveto(600, 210, 659, 305);
    curveto(718, 400, 746.5, 557);
    curveto(775, 714, 775, 784);
  }
  void add_bracketleft() {
    moveto(410, -425);
    lineto(410, 1484);
    lineto(957, 1484);
    lineto(957, 1345);
    lineto(590, 1345);
    lineto(590, -286);
    lineto(957, -286);
    lineto(957, -425);
    lineto(410, -425);
  }
  void add_backslash() {
    moveto(932, -20);
    lineto(115, 1484);
    lineto(293, 1484);
    lineto(1114, -20);
    lineto(932, -20);
  }
  void add_bracketright() {
    moveto(270, -425);
    lineto(270, -286);
    lineto(637, -286);
    lineto(637, 1345);
    lineto(270, 1345);
    lineto(270, 1484);
    lineto(817, 1484);
    lineto(817, -425);
    lineto(270, -425);
  }
  void add_asciicircum() {
    moveto(940, 442);
    lineto(611, 1245);
    lineto(285, 442);
    lineto(133, 442);
    lineto(511, 1349);
    lineto(714, 1349);
    lineto(1094, 442);
    lineto(940, 442);
  }
  void add_underscore() {
    moveto(-5, -220);
    lineto(-5, -124);
    lineto(1233, -124);
    lineto(1233, -220);
    lineto(-5, -220);
  }
  void add_grave() {
    moveto(702, 1201);
    lineto(402, 1431);
    lineto(402, 1460);
    lineto(599, 1460);
    lineto(826, 1221);
    lineto(826, 1201);
    lineto(702, 1201);
  }
  void add_a() {
    moveto(1101, 111);
    curveto(1127, 111, 1160, 118);
    lineto(1160, 6);
    curveto(1092, -10, 1021, -10);
    curveto(921, -10, 875.5, 42.5);
    curveto(830, 95, 824, 207);
    lineto(818, 207);
    curveto(753, 86, 664.5, 33);
    curveto(576, -20, 446, -20);
    curveto(288, -20, 208, 66);
    curveto(128, 152, 128, 302);
    curveto(128, 651, 582, 656);
    lineto(818, 660);
    lineto(818, 719);
    curveto(818, 850, 765, 907.5);
    curveto(712, 965, 596, 965);
    curveto(478, 965, 426, 923);
    curveto(374, 881, 364, 793);
    lineto(176, 810);
    curveto(222, 1102, 599, 1102);
    curveto(799, 1102, 899.5, 1008.5);
    curveto(1000, 915, 1000, 738);
    lineto(1000, 272);
    curveto(1000, 192, 1021, 151.5);
    curveto(1042, 111, 1101, 111);
    moveto(492, 117);
    curveto(588, 117, 662, 163);
    curveto(736, 209, 777, 286);
    curveto(818, 363, 818, 445);
    lineto(818, 534);
    lineto(628, 530);
    curveto(510, 528, 448, 504);
    curveto(386, 480, 351.5, 430.5);
    curveto(317, 381, 317, 299);
    curveto(317, 217, 361.5, 167);
    curveto(406, 117, 492, 117);
  }
  void add_b() {
    moveto(1090, 546);
    curveto(1090, 262, 988.5, 121);
    curveto(887, -20, 698, -20);
    curveto(454, -20, 364, 164);
    lineto(362, 164);
    curveto(362, 116, 358.5, 64);
    curveto(355, 12, 353, 0);
    lineto(179, 0);
    curveto(185, 54, 185, 223);
    lineto(185, 1484);
    lineto(365, 1484);
    lineto(365, 1061);
    curveto(365, 996, 361, 904);
    lineto(365, 904);
    curveto(456, 1104, 699, 1104);
    curveto(1090, 1104, 1090, 546);
    moveto(904, 540);
    curveto(904, 764, 842.5, 864.5);
    curveto(781, 965, 650, 965);
    curveto(501, 965, 433, 855.5);
    curveto(365, 746, 365, 524);
    curveto(365, 315, 431, 214);
    curveto(497, 113, 648, 113);
    curveto(783, 113, 843.5, 217.5);
    curveto(904, 322, 904, 540);
  }
  void add_c() {
    moveto(130, 542);
    curveto(130, 812, 259, 957);
    curveto(388, 1102, 632, 1102);
    curveto(814, 1102, 932, 1014.5);
    curveto(1050, 927, 1078, 779);
    lineto(886, 765);
    curveto(870, 856, 806, 908.5);
    curveto(742, 961, 624, 961);
    curveto(466, 961, 392.5, 863);
    curveto(319, 765, 319, 546);
    curveto(319, 324, 392.5, 221.5);
    curveto(466, 119, 623, 119);
    curveto(731, 119, 802, 172);
    curveto(873, 225, 890, 334);
    lineto(1080, 322);
    curveto(1067, 226, 1007.5, 147.5);
    curveto(948, 69, 850, 24.5);
    curveto(752, -20, 631, -20);
    curveto(386, -20, 258, 124);
    curveto(130, 268, 130, 542);
  }
  void add_d() {
    moveto(862, 174);
    curveto(813, 69, 732, 21.5);
    curveto(651, -26, 530, -26);
    curveto(328, -26, 233, 113);
    curveto(138, 252, 138, 532);
    curveto(138, 1098, 530, 1098);
    curveto(651, 1098, 732.5, 1055);
    curveto(814, 1012, 863, 914);
    lineto(865, 914);
    lineto(863, 1065);
    lineto(863, 1484);
    lineto(1043, 1484);
    lineto(1043, 223);
    curveto(1043, 54, 1049, 0);
    lineto(877, 0);
    curveto(873, 15, 870, 73);
    curveto(867, 131, 867, 174);
    lineto(862, 174);
    moveto(324, 538);
    curveto(324, 316, 383.5, 214.5);
    curveto(443, 113, 577, 113);
    curveto(723, 113, 793, 218.5);
    curveto(863, 324, 863, 554);
    curveto(863, 769, 795.5, 867);
    curveto(728, 965, 579, 965);
    curveto(444, 965, 384, 862);
    curveto(324, 759, 324, 538);
  }
  void add_e() {
    moveto(322, 503);
    curveto(322, 321, 402.5, 218);
    curveto(483, 115, 623, 115);
    curveto(726, 115, 803.5, 159.5);
    curveto(881, 204, 907, 281);
    lineto(1065, 236);
    curveto(1021, 112, 903.5, 46);
    curveto(786, -20, 623, -20);
    curveto(387, -20, 260, 127);
    curveto(133, 274, 133, 548);
    curveto(133, 815, 257.5, 958.5);
    curveto(382, 1102, 617, 1102);
    curveto(852, 1102, 973, 959);
    curveto(1094, 816, 1094, 527);
    lineto(1094, 503);
    lineto(322, 503);
    moveto(619, 969);
    curveto(485, 969, 407, 881.5);
    curveto(329, 794, 324, 641);
    lineto(908, 641);
    curveto(880, 969, 619, 969);
  }
  void add_f() {
    moveto(580, 940);
    lineto(580, 0);
    lineto(400, 0);
    lineto(400, 940);
    lineto(138, 940);
    lineto(138, 1082);
    lineto(400, 1082);
    lineto(400, 1107);
    curveto(400, 1312, 496.5, 1398);
    curveto(593, 1484, 818, 1484);
    curveto(890, 1484, 973.5, 1477.5);
    curveto(1057, 1471, 1099, 1463);
    lineto(1099, 1318);
    curveto(1069, 1323, 977.5, 1329);
    curveto(886, 1335, 839, 1335);
    curveto(735, 1335, 682, 1312);
    curveto(629, 1289, 604.5, 1237);
    curveto(580, 1185, 580, 1092);
    lineto(580, 1082);
    lineto(1071, 1082);
    lineto(1071, 940);
    lineto(580, 940);
  }
  void add_g() {
    moveto(615, -424);
    curveto(447, -424, 345, -355);
    curveto(243, -286, 215, -157);
    lineto(399, -132);
    curveto(416, -207, 472.5, -247.5);
    curveto(529, -288, 621, -288);
    curveto(869, -288, 869, 27);
    lineto(869, 221);
    lineto(867, 221);
    curveto(818, 118, 731, 65);
    curveto(644, 12, 524, 12);
    curveto(326, 12, 234.5, 141.5);
    curveto(143, 271, 143, 549);
    curveto(143, 832, 241, 965.5);
    curveto(339, 1099, 543, 1099);
    curveto(656, 1099, 739.5, 1046.5);
    curveto(823, 994, 868, 897);
    lineto(871, 897);
    curveto(871, 927, 875, 998.5);
    curveto(879, 1070, 883, 1082);
    lineto(1054, 1082);
    curveto(1048, 1028, 1048, 858);
    lineto(1048, 32);
    curveto(1048, -195, 942, -309.5);
    curveto(836, -424, 615, -424);
    moveto(869, 551);
    curveto(869, 744, 794, 854.5);
    curveto(719, 965, 588, 965);
    curveto(451, 965, 390, 869.5);
    curveto(329, 774, 329, 551);
    curveto(329, 400, 354.5, 312.5);
    curveto(380, 225, 434, 185);
    curveto(488, 145, 585, 145);
    curveto(670, 145, 734.5, 192.5);
    curveto(799, 240, 834, 331.5);
    curveto(869, 423, 869, 551);
  }
  void add_h() {
    moveto(185, 1484);
    lineto(366, 1484);
    lineto(366, 1094);
    curveto(366, 1035, 357, 897);
    lineto(360, 897);
    curveto(465, 1102, 699, 1102);
    curveto(1049, 1102, 1049, 721);
    lineto(1049, 0);
    lineto(868, 0);
    lineto(868, 695);
    curveto(868, 831, 815.5, 897);
    curveto(763, 963, 648, 963);
    curveto(524, 963, 444.5, 872.5);
    curveto(365, 782, 365, 627);
    lineto(365, 0);
    lineto(185, 0);
    lineto(185, 1484);
  }
  void add_i() {
    moveto(745, 142);
    lineto(1125, 142);
    lineto(1125, 0);
    lineto(143, 0);
    lineto(143, 142);
    lineto(565, 142);
    lineto(565, 940);
    lineto(246, 940);
    lineto(246, 1082);
    lineto(745, 1082);
    lineto(745, 142);
    moveto(545, 1292);
    lineto(545, 1484);
    lineto(745, 1484);
    lineto(745, 1292);
    lineto(545, 1292);
  }
  void add_j() {
    moveto(836, -28);
    curveto(836, -215, 724, -320);
    curveto(612, -425, 405, -425);
    curveto(326, -425, 244, -412);
    curveto(162, -399, 117, -382);
    lineto(117, -242);
    curveto(267, -276, 390, -276);
    curveto(518, -276, 587, -210);
    curveto(656, -144, 656, -25);
    lineto(656, 940);
    lineto(249, 940);
    lineto(249, 1082);
    lineto(836, 1082);
    lineto(836, -28);
    moveto(636, 1292);
    lineto(636, 1484);
    lineto(836, 1484);
    lineto(836, 1292);
    lineto(636, 1292);
  }
  void add_k() {
    moveto(914, 0);
    lineto(548, 499);
    lineto(416, 401);
    lineto(416, 0);
    lineto(236, 0);
    lineto(236, 1484);
    lineto(416, 1484);
    lineto(416, 557);
    lineto(891, 1082);
    lineto(1102, 1082);
    lineto(663, 617);
    lineto(1125, 0);
    lineto(914, 0);
  }
  void add_l() {
    moveto(835, 147);
    lineto(1116, 142);
    lineto(1116, 0);
    lineto(746, 4);
    curveto(633, 24, 581, 94);
    curveto(559, 124, 556, 258);
    lineto(556, 1342);
    lineto(267, 1342);
    lineto(267, 1484);
    lineto(736, 1484);
    lineto(736, 237);
    curveto(737, 192, 761, 170);
    curveto(782, 151, 835, 147);
    moveto(839, 142);
    moveto(736, 237);
    moveto(752, 0);
    moveto(556, 142);
    lineto(556, 258);
  }
  void add_m() {
    moveto(531, 0);
    lineto(531, 686);
    curveto(531, 840, 506.5, 901.5);
    curveto(482, 963, 417, 963);
    curveto(353, 963, 313.5, 867);
    curveto(274, 771, 274, 607);
    lineto(274, 0);
    lineto(105, 0);
    lineto(105, 851);
    curveto(105, 1040, 99, 1082);
    lineto(248, 1082);
    lineto(254, 955);
    lineto(254, 907);
    lineto(256, 907);
    curveto(290, 1009, 342, 1055.5);
    curveto(394, 1102, 472, 1102);
    curveto(560, 1102, 603.5, 1054);
    curveto(647, 1006, 666, 906);
    lineto(668, 906);
    curveto(708, 1012, 763.5, 1057);
    curveto(819, 1102, 904, 1102);
    curveto(1022, 1102, 1073, 1016);
    curveto(1124, 930, 1124, 721);
    lineto(1124, 0);
    lineto(956, 0);
    lineto(956, 686);
    curveto(956, 840, 931.5, 901.5);
    curveto(907, 963, 842, 963);
    curveto(776, 963, 737.5, 879);
    curveto(699, 795, 699, 627);
    lineto(699, 0);
    lineto(531, 0);
  }
  void add_n() {
    moveto(868, 0);
    lineto(868, 695);
    curveto(868, 831, 815.5, 897);
    curveto(763, 963, 648, 963);
    curveto(524, 963, 444.5, 872.5);
    curveto(365, 782, 365, 627);
    lineto(365, 0);
    lineto(185, 0);
    lineto(185, 851);
    curveto(185, 1040, 179, 1082);
    lineto(349, 1082);
    curveto(350, 1077, 351, 1055);
    curveto(352, 1033, 353.5, 1004.5);
    curveto(355, 976, 357, 897);
    lineto(360, 897);
    curveto(465, 1102, 706, 1102);
    curveto(879, 1102, 964, 1008.5);
    curveto(1049, 915, 1049, 721);
    lineto(1049, 0);
    lineto(868, 0);
  }
  void add_o() {
    moveto(1097, 542);
    curveto(1097, 269, 971.5, 124.5);
    curveto(846, -20, 609, -20);
    curveto(377, -20, 253.5, 126);
    curveto(130, 272, 130, 542);
    curveto(130, 821, 256.5, 961.5);
    curveto(383, 1102, 615, 1102);
    curveto(859, 1102, 978, 963);
    curveto(1097, 824, 1097, 542);
    moveto(908, 542);
    curveto(908, 757, 839.5, 863);
    curveto(771, 969, 618, 969);
    curveto(463, 969, 391, 861);
    curveto(319, 753, 319, 542);
    curveto(319, 332, 391, 222.5);
    curveto(463, 113, 607, 113);
    curveto(766, 113, 837, 220);
    curveto(908, 327, 908, 542);
  }
  void add_p() {
    moveto(1090, 546);
    curveto(1090, -20, 698, -20);
    curveto(452, -20, 367, 164);
    lineto(362, 164);
    curveto(366, 156, 366, -2);
    lineto(366, -425);
    lineto(185, -425);
    lineto(185, 858);
    curveto(185, 1028, 179, 1082);
    lineto(354, 1082);
    curveto(355, 1078, 357, 1052.5);
    curveto(359, 1027, 361.5, 977.5);
    curveto(364, 928, 364, 904);
    lineto(368, 904);
    curveto(418, 1009, 495.5, 1056.5);
    curveto(573, 1104, 698, 1104);
    curveto(896, 1104, 993, 967.5);
    curveto(1090, 831, 1090, 546);
    moveto(904, 546);
    curveto(904, 772, 843, 868.5);
    curveto(782, 965, 651, 965);
    curveto(500, 965, 433, 855.5);
    curveto(366, 746, 366, 524);
    curveto(366, 311, 433, 212);
    curveto(500, 113, 649, 113);
    curveto(782, 113, 843, 214);
    curveto(904, 315, 904, 546);
  }
  void add_q() {
    moveto(529, 1098);
    curveto(658, 1098, 735, 1053.5);
    curveto(812, 1009, 861, 914);
    lineto(863, 914);
    curveto(863, 944, 866.5, 1007.5);
    curveto(870, 1071, 875, 1083);
    lineto(1050, 1083);
    curveto(1044, 1029, 1044, 801);
    lineto(1044, -425);
    lineto(863, -425);
    lineto(863, 14);
    lineto(867, 182);
    lineto(865, 182);
    curveto(812, 75, 733, 24.5);
    curveto(654, -26, 530, -26);
    curveto(328, -26, 233, 114.5);
    curveto(138, 255, 138, 532);
    curveto(138, 1098, 529, 1098);
    moveto(863, 554);
    curveto(863, 766, 795, 865.5);
    curveto(727, 965, 579, 965);
    curveto(445, 965, 384.5, 863);
    curveto(324, 761, 324, 538);
    curveto(324, 317, 385, 215);
    curveto(446, 113, 577, 113);
    curveto(724, 113, 793.5, 222);
    curveto(863, 331, 863, 554);
  }
  void add_r() {
    moveto(1045, 918);
    curveto(933, 937, 833, 937);
    curveto(672, 937, 573, 816);
    curveto(474, 695, 474, 508);
    lineto(474, 0);
    lineto(294, 0);
    lineto(294, 701);
    curveto(294, 777, 280.5, 880);
    curveto(267, 983, 242, 1082);
    lineto(413, 1082);
    curveto(453, 944, 461, 832);
    lineto(466, 832);
    curveto(516, 944, 564, 996.5);
    curveto(612, 1049, 678, 1075.5);
    curveto(744, 1102, 839, 1102);
    curveto(943, 1102, 1045, 1085);
    lineto(1045, 918);
  }
  void add_s() {
    moveto(1060, 309);
    curveto(1060, 155, 943.5, 67.5);
    curveto(827, -20, 621, -20);
    curveto(415, -20, 307.5, 44.5);
    curveto(200, 109, 167, 248);
    lineto(326, 279);
    curveto(345, 193, 407.5, 153.5);
    curveto(470, 114, 621, 114);
    curveto(891, 114, 891, 285);
    curveto(891, 349, 842, 388.5);
    curveto(793, 428, 692, 453);
    curveto(428, 518, 357, 555);
    curveto(286, 592, 248, 647.5);
    curveto(210, 703, 210, 786);
    curveto(210, 933, 316, 1016);
    curveto(422, 1099, 623, 1099);
    curveto(799, 1099, 904, 1032.5);
    curveto(1009, 966, 1035, 839);
    lineto(873, 819);
    curveto(862, 891, 802, 928);
    curveto(742, 965, 623, 965);
    curveto(378, 965, 378, 814);
    curveto(378, 754, 419.5, 718);
    curveto(461, 682, 553, 660);
    lineto(672, 629);
    curveto(835, 589, 906.5, 550);
    curveto(978, 511, 1019, 452.5);
    curveto(1060, 394, 1060, 309);
  }
  void add_t() {
    moveto(190, 940);
    lineto(190, 1082);
    lineto(360, 1082);
    lineto(418, 1364);
    lineto(538, 1364);
    lineto(538, 1082);
    lineto(970, 1082);
    lineto(970, 940);
    lineto(538, 940);
    lineto(538, 288);
    curveto(538, 209, 580.5, 171);
    curveto(623, 133, 720, 133);
    curveto(854, 133, 1017, 167);
    lineto(1017, 30);
    curveto(848, -16, 682, -16);
    curveto(520, -16, 439, 52.5);
    curveto(358, 121, 358, 269);
    lineto(358, 940);
    lineto(190, 940);
  }
  void add_u() {
    moveto(365, 1082);
    lineto(365, 396);
    curveto(365, 240, 414, 179.5);
    curveto(463, 119, 589, 119);
    curveto(718, 119, 793, 207);
    curveto(868, 295, 868, 455);
    lineto(868, 1082);
    lineto(1049, 1082);
    lineto(1049, 231);
    curveto(1049, 42, 1055, 0);
    lineto(885, 0);
    curveto(884, 5, 883, 27);
    curveto(882, 49, 880.5, 77.5);
    curveto(879, 106, 877, 185);
    lineto(874, 185);
    curveto(812, 73, 730.5, 26.5);
    curveto(649, -20, 528, -20);
    curveto(350, -20, 267.5, 68.5);
    curveto(185, 157, 185, 361);
    lineto(185, 1082);
    lineto(365, 1082);
  }
  void add_v() {
    moveto(715, 0);
    lineto(502, 0);
    lineto(69, 1082);
    lineto(271, 1082);
    lineto(539, 378);
    lineto(556, 325);
    lineto(608, 141);
    lineto(643, 258);
    lineto(682, 376);
    lineto(958, 1082);
    lineto(1159, 1082);
    lineto(715, 0);
  }
  void add_w() {
    moveto(1018, 0);
    lineto(814, 0);
    lineto(671, 471);
    lineto(614, 673);
    lineto(575, 534);
    lineto(407, 0);
    lineto(204, 0);
    lineto(21, 1082);
    lineto(199, 1082);
    lineto(292, 475);
    curveto(325, 211, 325, 149);
    curveto(364, 309, 383, 363);
    lineto(518, 787);
    lineto(711, 787);
    lineto(841, 362);
    curveto(873, 257, 896, 149);
    curveto(896, 179, 900, 224);
    curveto(904, 269, 910, 316.5);
    curveto(916, 364, 922, 407);
    curveto(928, 450, 931, 475);
    lineto(1032, 1082);
    lineto(1208, 1082);
    lineto(1018, 0);
  }
  void add_x() {
    moveto(932, 0);
    lineto(611, 444);
    lineto(288, 0);
    lineto(94, 0);
    lineto(509, 556);
    lineto(112, 1082);
    lineto(311, 1082);
    lineto(611, 661);
    lineto(909, 1082);
    lineto(1110, 1082);
    lineto(713, 558);
    lineto(1133, 0);
    lineto(932, 0);
  }
  void add_y() {
    moveto(292, -425);
    curveto(218, -425, 168, -414);
    lineto(168, -279);
    curveto(206, -285, 252, -285);
    curveto(331, -285, 400.5, -226);
    curveto(470, -167, 518, -38);
    lineto(536, 11);
    lineto(66, 1082);
    lineto(258, 1082);
    lineto(522, 440);
    curveto(613, 216, 620, 186);
    lineto(661, 296);
    lineto(971, 1082);
    lineto(1161, 1082);
    lineto(705, 0);
    curveto(616, -235, 520.5, -330);
    curveto(425, -425, 292, -425);
  }
  void add_z() {
    moveto(147, 0);
    lineto(147, 137);
    lineto(828, 943);
    lineto(187, 943);
    lineto(187, 1082);
    lineto(1031, 1082);
    lineto(1031, 945);
    lineto(349, 139);
    lineto(1068, 139);
    lineto(1068, 0);
    lineto(147, 0);
  }
  void add_braceleft() {
    moveto(796, -425);
    curveto(666, -425, 584, -340.5);
    curveto(502, -256, 502, -122);
    lineto(502, 229);
    curveto(502, 336, 429, 396);
    curveto(356, 456, 227, 461);
    lineto(227, 598);
    curveto(357, 603, 429.5, 664);
    curveto(502, 725, 502, 829);
    lineto(502, 1181);
    curveto(502, 1318, 583, 1401);
    curveto(664, 1484, 796, 1484);
    lineto(1061, 1484);
    lineto(1061, 1345);
    lineto(848, 1345);
    curveto(757, 1345, 717, 1301);
    curveto(677, 1257, 677, 1150);
    lineto(677, 804);
    curveto(677, 706, 613, 631.5);
    curveto(549, 557, 446, 532);
    lineto(446, 530);
    curveto(548, 506, 612.5, 432.5);
    curveto(677, 359, 677, 256);
    lineto(677, -91);
    curveto(677, -197, 717, -241.5);
    curveto(757, -286, 848, -286);
    lineto(1061, -286);
    lineto(1061, -425);
    lineto(796, -425);
  }
  void add_bar() {
    moveto(531, -425);
    lineto(531, 1484);
    lineto(697, 1484);
    lineto(697, -425);
    lineto(531, -425);
  }
  void add_braceright() {
    moveto(167, -425);
    lineto(167, -286);
    lineto(380, -286);
    curveto(471, -286, 511.5, -241.5);
    curveto(552, -197, 552, -91);
    lineto(552, 256);
    curveto(552, 358, 616, 432);
    curveto(680, 506, 782, 530);
    lineto(782, 532);
    curveto(680, 557, 616, 632);
    curveto(552, 707, 552, 804);
    lineto(552, 1150);
    curveto(552, 1255, 512, 1300);
    curveto(472, 1345, 380, 1345);
    lineto(167, 1345);
    lineto(167, 1484);
    lineto(432, 1484);
    curveto(565, 1484, 645.5, 1401);
    curveto(726, 1318, 726, 1181);
    lineto(726, 829);
    curveto(726, 724, 799, 663.5);
    curveto(872, 603, 1001, 598);
    lineto(1001, 461);
    curveto(871, 456, 798.5, 395.5);
    curveto(726, 335, 726, 229);
    lineto(726, -122);
    curveto(726, -257, 644, -341);
    curveto(562, -425, 432, -425);
    lineto(167, -425);
  }
  void add_asciitilde() {
    moveto(371, 807);
    curveto(473, 807, 606, 761);
    curveto(753, 710, 796.5, 700);
    curveto(840, 690, 876, 690);
    curveto(1006, 690, 1120, 782);
    lineto(1120, 633);
    curveto(1060, 591, 1002.5, 572);
    curveto(945, 553, 860, 553);
    curveto(791, 553, 718.5, 575);
    curveto(646, 597, 573, 623);
    curveto(444, 668, 356, 668);
    curveto(289, 668, 231, 647.5);
    curveto(173, 627, 108, 580);
    lineto(108, 723);
    curveto(219, 807, 371, 807);
  }
  void add_A() {
    moveto(1034, 0);
    lineto(896, 382);
    lineto(333, 382);
    lineto(196, 0);
    lineto(0, 0);
    lineto(510, 1349);
    lineto(727, 1349);
    lineto(1228, 0);
    lineto(1034, 0);
    moveto(616, 1205);
    lineto(604, 1166);
    lineto(535, 954);
    lineto(384, 531);
    lineto(847, 531);
    lineto(674, 1031);
    lineto(616, 1205);
  }
  void add_B() {
    moveto(1152, 380);
    curveto(1152, 200, 1015, 100);
    curveto(878, 0, 634, 0);
    lineto(162, 0);
    lineto(162, 1349);
    lineto(574, 1349);
    curveto(1070, 1349, 1070, 1022);
    curveto(1070, 900, 998, 818);
    curveto(926, 736, 802, 711);
    curveto(969, 693, 1060.5, 604);
    curveto(1152, 515, 1152, 380);
    moveto(878, 998);
    curveto(878, 1105, 802.5, 1150.5);
    curveto(727, 1196, 576, 1196);
    lineto(353, 1196);
    lineto(353, 780);
    lineto(578, 780);
    curveto(878, 780, 878, 998);
    moveto(959, 397);
    curveto(959, 511, 870, 571);
    curveto(781, 631, 605, 631);
    lineto(353, 631);
    lineto(353, 153);
    lineto(619, 153);
    curveto(794, 153, 876.5, 213.5);
    curveto(959, 274, 959, 397);
  }
  void add_C() {
    moveto(314, 681);
    curveto(314, 408, 399.5, 271.5);
    curveto(485, 135, 661, 135);
    curveto(762, 135, 844, 203.5);
    curveto(926, 272, 983, 417);
    lineto(1142, 352);
    curveto(993, -20, 659, -20);
    curveto(396, -20, 254.5, 161);
    curveto(113, 342, 113, 681);
    curveto(113, 1370, 649, 1370);
    curveto(988, 1370, 1115, 1035);
    lineto(947, 970);
    curveto(910, 1083, 831.5, 1148.5);
    curveto(753, 1214, 650, 1214);
    curveto(479, 1214, 396.5, 1085);
    curveto(314, 956, 314, 681);
  }
  void add_D() {
    moveto(1125, 688);
    curveto(1125, 357, 971.5, 178.5);
    curveto(818, 0, 532, 0);
    lineto(162, 0);
    lineto(162, 1349);
    lineto(473, 1349);
    curveto(802, 1349, 963.5, 1184.5);
    curveto(1125, 1020, 1125, 688);
    moveto(933, 688);
    curveto(933, 952, 823, 1072.5);
    curveto(713, 1193, 474, 1193);
    lineto(353, 1193);
    lineto(353, 156);
    lineto(515, 156);
    curveto(727, 156, 830, 289);
    curveto(933, 422, 933, 688);
  }
  void add_E() {
    moveto(162, 0);
    lineto(162, 1349);
    lineto(1081, 1349);
    lineto(1081, 1193);
    lineto(353, 1193);
    lineto(353, 771);
    lineto(1021, 771);
    lineto(1021, 617);
    lineto(353, 617);
    lineto(353, 156);
    lineto(1122, 156);
    lineto(1122, 0);
    lineto(162, 0);
  }
  void add_F() {
    moveto(385, 1193);
    lineto(385, 699);
    lineto(1061, 699);
    lineto(1061, 541);
    lineto(385, 541);
    lineto(385, 0);
    lineto(194, 0);
    lineto(194, 1349);
    lineto(1085, 1349);
    lineto(1085, 1193);
    lineto(385, 1193);
  }
  void add_G() {
    moveto(1101, 133);
    curveto(872, -20, 639, -20);
    curveto(389, -20, 251, 165.5);
    curveto(113, 351, 113, 681);
    curveto(113, 1028, 245, 1199);
    curveto(377, 1370, 642, 1370);
    curveto(988, 1370, 1103, 1039);
    lineto(932, 983);
    curveto(852, 1214, 644, 1214);
    curveto(475, 1214, 394.5, 1088);
    curveto(314, 962, 314, 681);
    curveto(314, 135, 655, 135);
    curveto(723, 135, 796, 156);
    curveto(869, 177, 915, 209);
    lineto(915, 545);
    lineto(622, 545);
    lineto(622, 705);
    lineto(1101, 705);
    lineto(1101, 133);
  }
  void add_H() {
    moveto(875, 0);
    lineto(875, 623);
    lineto(353, 623);
    lineto(353, 0);
    lineto(162, 0);
    lineto(162, 1349);
    lineto(353, 1349);
    lineto(353, 783);
    lineto(875, 783);
    lineto(875, 1349);
    lineto(1066, 1349);
    lineto(1066, 0);
    lineto(875, 0);
  }
  void add_I() {
    moveto(202, 1349);
    lineto(1025, 1349);
    lineto(1025, 1193);
    lineto(709, 1193);
    lineto(709, 156);
    lineto(1025, 156);
    lineto(1025, 0);
    lineto(202, 0);
    lineto(202, 156);
    lineto(518, 156);
    lineto(518, 1193);
    lineto(202, 1193);
    lineto(202, 1349);
  }
  void add_J() {
    moveto(986, 420);
    curveto(986, 211, 881, 95.5);
    curveto(776, -20, 586, -20);
    curveto(415, -20, 313, 69);
    curveto(211, 158, 176, 350);
    lineto(363, 381);
    curveto(381, 263, 439.5, 199);
    curveto(498, 135, 587, 135);
    curveto(796, 135, 796, 416);
    lineto(796, 1193);
    lineto(485, 1193);
    lineto(485, 1349);
    lineto(986, 1349);
    lineto(986, 420);
  }
  void add_K() {
    moveto(1003, 0);
    lineto(516, 638);
    lineto(353, 469);
    lineto(353, 0);
    lineto(162, 0);
    lineto(162, 1349);
    lineto(353, 1349);
    lineto(353, 676);
    lineto(925, 1349);
    lineto(1150, 1349);
    lineto(646, 777);
    lineto(1227, 0);
    lineto(1003, 0);
  }
  void add_L() {
    moveto(237, 0);
    lineto(237, 1349);
    lineto(428, 1349);
    lineto(428, 156);
    lineto(1100, 156);
    lineto(1100, 0);
    lineto(237, 0);
  }
  void add_M() {
    moveto(937, 0);
    lineto(937, 868);
    curveto(937, 1003, 940, 1069);
    lineto(943, 1169);
    curveto(879, 963, 848, 878);
    lineto(684, 440);
    lineto(547, 440);
    lineto(381, 878);
    curveto(363, 924, 285, 1169);
    lineto(289, 868);
    lineto(289, 0);
    lineto(129, 0);
    lineto(129, 1349);
    lineto(366, 1349);
    lineto(551, 860);
    curveto(572, 807, 619, 629);
    lineto(645, 719);
    lineto(689, 859);
    lineto(874, 1349);
    lineto(1099, 1349);
    lineto(1099, 0);
    lineto(937, 0);
  }
  void add_N() {
    moveto(836, 0);
    lineto(316, 1130);
    curveto(332, 958, 332, 876);
    lineto(332, 0);
    lineto(162, 0);
    lineto(162, 1349);
    lineto(384, 1349);
    lineto(912, 211);
    curveto(894, 355, 894, 485);
    lineto(894, 1349);
    lineto(1066, 1349);
    lineto(1066, 0);
    lineto(836, 0);
  }
  void add_O() {
    moveto(1126, 681);
    curveto(1126, 344, 993.5, 162);
    curveto(861, -20, 613, -20);
    curveto(364, -20, 233, 159);
    curveto(102, 338, 102, 681);
    curveto(102, 1018, 232, 1194);
    curveto(362, 1370, 615, 1370);
    curveto(862, 1370, 994, 1196.5);
    curveto(1126, 1023, 1126, 681);
    moveto(925, 681);
    curveto(925, 1214, 615, 1214);
    curveto(303, 1214, 303, 681);
    curveto(303, 411, 382, 273);
    curveto(461, 135, 614, 135);
    curveto(777, 135, 851, 275);
    curveto(925, 415, 925, 681);
  }
  void add_P() {
    moveto(1119, 945);
    curveto(1119, 820, 1059.5, 722);
    curveto(1000, 624, 889.5, 569);
    curveto(779, 514, 634, 514);
    lineto(353, 514);
    lineto(353, 0);
    lineto(162, 0);
    lineto(162, 1349);
    lineto(622, 1349);
    curveto(860, 1349, 989.5, 1242.5);
    curveto(1119, 1136, 1119, 945);
    moveto(927, 942);
    curveto(927, 1196, 599, 1196);
    lineto(353, 1196);
    lineto(353, 665);
    lineto(607, 665);
    curveto(756, 665, 841.5, 738);
    curveto(927, 811, 927, 942);
  }
  void add_Q() {
    moveto(1126, 681);
    curveto(1126, 391, 1027.5, 215.5);
    curveto(929, 40, 746, -4);
    curveto(787, -130, 854, -187);
    curveto(921, -244, 1022, -244);
    curveto(1077, -244, 1137, -231);
    lineto(1137, -365);
    curveto(1044, -387, 959, -387);
    curveto(809, -387, 711.5, -302.5);
    curveto(614, -218, 551, -16);
    curveto(332, 6, 217, 183.5);
    curveto(102, 361, 102, 681);
    curveto(102, 1018, 232, 1194);
    curveto(362, 1370, 615, 1370);
    curveto(862, 1370, 994, 1196.5);
    curveto(1126, 1023, 1126, 681);
    moveto(925, 681);
    curveto(925, 1214, 615, 1214);
    curveto(303, 1214, 303, 681);
    curveto(303, 411, 382, 273);
    curveto(461, 135, 614, 135);
    curveto(777, 135, 851, 275);
    curveto(925, 415, 925, 681);
  }
  void add_R() {
    moveto(957, 0);
    lineto(591, 575);
    lineto(353, 575);
    lineto(353, 0);
    lineto(162, 0);
    lineto(162, 1349);
    lineto(644, 1349);
    curveto(877, 1349, 999, 1252.5);
    curveto(1121, 1156, 1121, 976);
    curveto(1121, 827, 1027.5, 725);
    curveto(934, 623, 777, 597);
    lineto(1177, 0);
    lineto(957, 0);
    moveto(929, 973);
    curveto(929, 1196, 625, 1196);
    lineto(353, 1196);
    lineto(353, 726);
    lineto(633, 726);
    curveto(776, 726, 852.5, 790);
    curveto(929, 854, 929, 973);
  }
  void add_S() {
    moveto(1128, 370);
    curveto(1128, 186, 993.5, 83);
    curveto(859, -20, 610, -20);
    curveto(153, -20, 79, 338);
    lineto(264, 375);
    curveto(292, 246, 380, 187.5);
    curveto(468, 129, 615, 129);
    curveto(774, 129, 856.5, 191);
    curveto(939, 253, 939, 367);
    curveto(939, 437, 906.5, 481);
    curveto(874, 525, 821, 553);
    curveto(768, 581, 701.5, 598.5);
    curveto(635, 616, 567, 633);
    curveto(406, 675, 337.5, 708);
    curveto(269, 741, 228, 783.5);
    curveto(187, 826, 166, 881);
    curveto(145, 936, 145, 1010);
    curveto(145, 1183, 266.5, 1276.5);
    curveto(388, 1370, 615, 1370);
    curveto(827, 1370, 939, 1296);
    curveto(1051, 1222, 1095, 1046);
    lineto(907, 1013);
    curveto(883, 1125, 811, 1175.5);
    curveto(739, 1226, 614, 1226);
    curveto(331, 1226, 331, 1013);
    curveto(331, 953, 357.5, 915.5);
    curveto(384, 878, 429.5, 853.5);
    curveto(475, 829, 535.5, 813);
    curveto(596, 797, 665, 779);
    curveto(804, 744, 865, 720.5);
    curveto(926, 697, 973.5, 667);
    curveto(1021, 637, 1055, 596);
    curveto(1089, 555, 1108.5, 500);
    curveto(1128, 445, 1128, 370);
  }
  void add_T() {
    moveto(709, 1193);
    lineto(709, 0);
    lineto(519, 0);
    lineto(519, 1193);
    lineto(76, 1193);
    lineto(76, 1349);
    lineto(1152, 1349);
    lineto(1152, 1193);
    lineto(709, 1193);
  }
  void add_U() {
    moveto(1085, 490);
    curveto(1085, 223, 971, 101.5);
    curveto(857, -20, 605, -20);
    curveto(361, -20, 251.5, 97.5);
    curveto(142, 215, 142, 472);
    lineto(142, 1349);
    lineto(333, 1349);
    lineto(333, 498);
    curveto(333, 294, 391.5, 214.5);
    curveto(450, 135, 604, 135);
    curveto(765, 135, 830, 217);
    curveto(895, 299, 895, 511);
    lineto(895, 1349);
    lineto(1085, 1349);
    lineto(1085, 490);
  }
  void add_V() {
    moveto(713, 0);
    lineto(515, 0);
    lineto(10, 1349);
    lineto(211, 1349);
    lineto(531, 447);
    curveto(562, 361, 615, 168);
    curveto(655, 317, 699, 447);
    lineto(1017, 1349);
    lineto(1218, 1349);
    lineto(713, 0);
  }
  void add_W() {
    moveto(1018, 0);
    lineto(810, 0);
    curveto(736, 276, 697.5, 420);
    curveto(659, 564, 616, 756);
    curveto(587, 631, 562.5, 530);
    curveto(538, 429, 419, 0);
    lineto(211, 0);
    lineto(0, 1349);
    lineto(189, 1349);
    lineto(298, 514);
    curveto(314, 383, 331, 168);
    curveto(363, 306, 384.5, 396);
    curveto(406, 486, 528, 931);
    lineto(703, 931);
    curveto(772, 678, 811.5, 533);
    curveto(851, 388, 900, 168);
    lineto(935, 514);
    lineto(1039, 1349);
    lineto(1228, 1349);
    lineto(1018, 0);
  }
  void add_X() {
    moveto(614, 836);
    lineto(947, 1349);
    lineto(1152, 1349);
    lineto(717, 705);
    lineto(1193, 0);
    lineto(988, 0);
    lineto(614, 573);
    lineto(241, 0);
    lineto(36, 0);
    lineto(512, 705);
    lineto(77, 1349);
    lineto(282, 1349);
    lineto(614, 836);
  }
  void add_Y() {
    moveto(708, 584);
    lineto(708, 0);
    lineto(520, 0);
    lineto(520, 584);
    lineto(36, 1349);
    lineto(241, 1349);
    lineto(615, 738);
    lineto(987, 1349);
    lineto(1192, 1349);
    lineto(708, 584);
  }
  void add_Z() {
    moveto(1155, 0);
    lineto(73, 0);
    lineto(73, 143);
    lineto(891, 1193);
    lineto(146, 1193);
    lineto(146, 1349);
    lineto(1108, 1349);
    lineto(1108, 1210);
    lineto(290, 156);
    lineto(1155, 156);
    lineto(1155, 0);
  }
  void add_char(char c) {
    m_outline_ptr->clear();
    switch(c) {
      case '!': add_exclam(); break;
      case '"': add_quotedbl(); break;
      case '#': add_numbersign(); break;
      case '$': add_dollar(); break;
      case '%': add_percent(); break;
      case '&': add_ampersand(); break;
      case '\'': add_quotesingle(); break;
      case '(': add_parenleft(); break;
      case ')': add_parenright(); break;
      case '*': add_asterisk(); break;
      case '+': add_plus(); break;
      case ',': add_comma(); break;
      case '-': add_hyphen(); break;
      case '.': add_period(); break;
      case '/': add_slash(); break;
      case '0': add_zero(); break;
      case '1': add_one(); break;
      case '2': add_two(); break;
      case '3': add_three(); break;
      case '4': add_four(); break;
      case '5': add_five(); break;
      case '6': add_six(); break;
      case '7': add_seven(); break;
      case '8': add_eight(); break;
      case '9': add_nine(); break;
      case ':': add_colon(); break;
      case ';': add_semicolon(); break;
      case '<': add_less(); break;
      case '=': add_equal(); break;
      case '>': add_greater(); break;
      case '?': add_question(); break;
      case '@': add_at(); break;
      case 'A': add_A(); break;
      case 'B': add_B(); break;
      case 'C': add_C(); break;
      case 'D': add_D(); break;
      case 'E': add_E(); break;
      case 'F': add_F(); break;
      case 'G': add_G(); break;
      case 'H': add_H(); break;
      case 'I': add_I(); break;
      case 'J': add_J(); break;
      case 'K': add_K(); break;
      case 'L': add_L(); break;
      case 'M': add_M(); break;
      case 'N': add_N(); break;
      case 'O': add_O(); break;
      case 'P': add_P(); break;
      case 'Q': add_Q(); break;
      case 'R': add_R(); break;
      case 'S': add_S(); break;
      case 'T': add_T(); break;
      case 'U': add_U(); break;
      case 'V': add_V(); break;
      case 'W': add_W(); break;
      case 'X': add_X(); break;
      case 'Y': add_Y(); break;
      case 'Z': add_Z(); break;
      case '[': add_bracketleft(); break;
      case '\\': add_backslash(); break;
      case ']': add_bracketright(); break;
      case '^': add_asciicircum(); break;
      case '_': add_underscore(); break;
      case '`': add_grave(); break;
      case 'a': add_a(); break;
      case 'b': add_b(); break;
      case 'c': add_c(); break;
      case 'd': add_d(); break;
      case 'e': add_e(); break;
      case 'f': add_f(); break;
      case 'g': add_g(); break;
      case 'h': add_h(); break;
      case 'i': add_i(); break;
      case 'j': add_j(); break;
      case 'k': add_k(); break;
      case 'l': add_l(); break;
      case 'm': add_m(); break;
      case 'n': add_n(); break;
      case 'o': add_o(); break;
      case 'p': add_p(); break;
      case 'q': add_q(); break;
      case 'r': add_r(); break;
      case 's': add_s(); break;
      case 't': add_t(); break;
      case 'u': add_u(); break;
      case 'v': add_v(); break;
      case 'w': add_w(); break;
      case 'x': add_x(); break;
      case 'y': add_y(); break;
      case 'z': add_z(); break;
      case '{': add_braceleft(); break;
      case '|': add_bar(); break;
      case '}': add_braceright(); break;
      case '~': add_asciitilde(); break;
    }
  }
};

class canvas {
  coloring::image m_image;
  coloring::outline m_outline;
 public:
  canvas(
      unsigned width,
      unsigned height,
      byte_rgba const& background)
    :m_image(width, height)
  {
    auto const pixel_box = m_image.pixel_box();
    coloring::for_each(pixel_box,
    [&] (coloring::int_vec const& p) {
      m_image.set(p, background);
    });
  }
  void draw_text(
      std::string const& text,
      float_vec const& anchor,
      text_anchor anchor_type,
      bool vertical,
      float width,
      byte_rgba const& color)
  {
    float const full_width = text.length() * width;
    float const half_full_width = full_width / 2.0f;
    float const full_height = this->text_height(width);
    float const half_full_height = full_height / 2.0f;
    float_vec offset;
    switch (anchor_type) {
      case text_anchor::top:
        offset = float_vec(half_full_width, full_height);
        break;
      case text_anchor::bottom:
        offset = float_vec(half_full_width, 0.0f);
        break;
      case text_anchor::left:
        offset = float_vec(0.0f, half_full_height);
        break;
      case text_anchor::right:
        offset = float_vec(full_width, half_full_height);
        break;
    }
    if (vertical) offset = rotate_left_90deg(offset);
    coloring::font font(m_outline, anchor - offset, width, vertical);
    for (char c : text) {
      font.add_char(c);
      draw(m_image, m_outline, color);
      font.advance();
    }
  }
  float text_height(float width)
  {
    coloring::font font(m_outline, {0,0}, width, false);
    return font.height();
  }
  void draw_line(
      float_vec const& a,
      float_vec const& b,
      byte_rgba const& color,
      float thickness)
  {
    if (a == b) {
      draw_point(a, color, thickness);
      return;
    }
    float_vec const ab = b - a;
    float_vec const v = ab * ((thickness / 2) / magnitude(ab));
    float_vec const b0 = b + rotate_left_90deg(v);
    float_vec const b01 = b + rotate_left_90deg(v) + v;
    float_vec const b1 = b + v;
    float_vec const b12 = b + v + rotate_right_90deg(v);
    float_vec const b2 = b + rotate_right_90deg(v);
    float_vec const a0 = a - rotate_left_90deg(v);
    float_vec const a01 = a - rotate_left_90deg(v) - v;
    float_vec const a1 = a - v;
    float_vec const a12 = a - v - rotate_right_90deg(v);
    float_vec const a2 = a - rotate_right_90deg(v);
    m_outline.clear();
    m_outline.add(line(a2, b0));
    m_outline.add(line(b2, a0));
    m_outline.add(quadratic_bezier(b0, b01, b1));
    m_outline.add(quadratic_bezier(b1, b12, b2));
    m_outline.add(quadratic_bezier(a0, a01, a1));
    m_outline.add(quadratic_bezier(a1, a12, a2));
    draw(m_image, m_outline, color);
  }
  void draw_box(
      float_box const& box,
      byte_rgba const& color)
  {
    m_outline.clear();
    m_outline.add(line(box.lower_left(), box.upper_left()));
    m_outline.add(line(box.upper_left(), box.upper_right()));
    m_outline.add(line(box.upper_right(), box.lower_right()));
    m_outline.add(line(box.lower_right(), box.lower_left()));
    draw(m_image, m_outline, color);
  }
  void draw_point(
      float_vec const& p,
      byte_rgba const& color,
      float thickness)
  {
    float_vec const v = float_vec(thickness / 2, 0);
    float_vec const top = p + rotate_left_90deg(v);
    float_vec const top_right = p + rotate_left_90deg(v) + v;
    float_vec const right = p + v;
    float_vec const bottom_right = p + v + rotate_right_90deg(v);
    float_vec const bottom = p + rotate_right_90deg(v);
    float_vec const bottom_left = p - rotate_left_90deg(v) - v;
    float_vec const left = p - v;
    float_vec const top_left = p - v - rotate_right_90deg(v);
    m_outline.clear();
    m_outline.add(quadratic_bezier(top, top_right, right));
    m_outline.add(quadratic_bezier(right, bottom_right, bottom));
    m_outline.add(quadratic_bezier(bottom, bottom_left, left));
    m_outline.add(quadratic_bezier(left, top_left, top));
    draw(m_image, m_outline, color);
  }
  float width() const { return m_image.width(); }
  float height() const { return m_image.height(); }
  void write(std::filesystem::path const& pngpath) const
  {
    m_image.write(pngpath);
  }
  ~canvas()
  {
  }
};

static void draw_box_outline(
    coloring::canvas& canvas,
    float_box const& box,
    byte_rgba const& color,
    float thickness)
{
  canvas.draw_line(
      box.lower_left(), box.lower_right(), color, thickness);
  canvas.draw_line(
      box.lower_right(), box.upper_right(), color, thickness);
  canvas.draw_line(
      box.upper_right(), box.upper_left(), color, thickness);
  canvas.draw_line(
      box.upper_left(), box.lower_left(), color, thickness);
}

inline constexpr float_vec data_to_view(
    float_vec p,
    float_box const& data_box,
    float_box const& view_box)
{
  float_vec const data_extents = data_box.extents();
  float_vec const view_extents = view_box.extents();
  p = p - data_box.lower();
  p = float_vec(p.x() / data_extents.x(), p.y() / data_extents.y());
  p = float_vec(p.x() * view_extents.x(), p.y() * view_extents.y());
  p = p + view_box.lower();
  return p;
}

inline void compute_ticks(
    float low,
    float high,
    std::vector<float>& ticks,
    int tick_count = 5)
{
  float const extent = high - low;
  float const extent_log10 = std::log10(extent);
  float const tick_count_log10 = std::log10(float(tick_count));
  int const grid_spacing_log10 = int(std::floor(extent_log10) - std::ceil(tick_count_log10));
  float const grid_spacing = std::pow(10.0f, grid_spacing_log10);
  int const grid_first = int(std::ceil(low / grid_spacing));
  int const grid_last = int(std::floor(high / grid_spacing));
  int const grid_extent = (grid_last - grid_first);
  ticks.clear();
  for (int tick_i = 0; tick_i < tick_count; ++tick_i) {
    int const grid_i = (tick_i * grid_extent) / (tick_count - 1) + grid_first;
    float const tick = grid_i * grid_spacing;
    ticks.push_back(tick);
  }
}

inline constexpr int plot_text_width = 10;
inline constexpr int plot_breathing_room = 6;
inline constexpr float default_line_thickness = 2;
inline constexpr float default_point_thickness = 4;
inline constexpr float plot_text_height = font::aspect_ratio * plot_text_width;
inline constexpr int tick_char_count = 9;
inline constexpr float tick_label_width =
    tick_char_count * plot_text_width;
inline constexpr int legend_line_length = 15;

static float_box compute_bounding_box(
    plot_group const& group, std::size_t dataset_index)
{
  auto& dataset = group.datasets[dataset_index];
  auto& x_array = dataset.x;
  auto& y_array = dataset.y;
  std::size_t n = x_array.size();
  if (y_array.size() != n) {
    throw std::runtime_error(
        "plot group "
        + group.name
        + " dataset "
        + std::to_string(dataset_index)
        + " x and y arrays are different sizes");
  }
  float_box result;
  for (std::size_t i = 0; i < n; ++i) {
    result.include(float_vec(x_array[i], y_array[i]));
  }
  return result;
}

static float_box compute_bounding_box(
    plot_group const& group)
{
  float_box result;
  for (std::size_t i = 0; i < group.datasets.size(); ++i) {
    result.include(compute_bounding_box(group, i));
  }
  return result;
}

static float_box compute_bounding_box(
    plot_data const& data)
{
  float_box result;
  for (auto& group : data.groups) {
    result.include(compute_bounding_box(group));
  }
  return result;
}

static int max_name_length(
    plot_data const& data)
{
  int result = 0;
  for (auto& group : data.groups) {
    result = max(result, int(group.name.length()));
  }
  return result;
}

static void draw_group(
    coloring::canvas& canvas,
    float_box const& view_box,
    float_box const& data_box,
    plot_group const& group)
{
  for (auto& dataset : group.datasets) {
    auto& x_array = dataset.x;
    auto& y_array = dataset.y;
    std::size_t const n = x_array.size();
    if (dataset.style == plot_style::lines) {
      float thickness = dataset.thickness.value_or(default_line_thickness);
      for (std::size_t i = 1; i < n; ++i) {
        float_vec const data_a(x_array[i - 1], y_array[i - 1]);
        float_vec const data_b(x_array[i], y_array[i]);
        float_vec const view_a = data_to_view(data_a, data_box, view_box);
        float_vec const view_b = data_to_view(data_b, data_box, view_box);
        canvas.draw_line(view_a, view_b,
            group.color.value(), thickness);
      }
    } else if (dataset.style == plot_style::points) {
      float thickness = dataset.thickness.value_or(default_point_thickness);
      for (std::size_t i = 0; i < n; ++i) {
        float_vec const data_point(x_array[i], y_array[i]);
        float_vec const view_point = data_to_view(data_point, data_box, view_box);
        canvas.draw_point(view_point,
            group.color.value(), thickness);
      }
    }
  }
}

static float_vec compute_legend_extents(
    plot_data const& data)
{
  int char_count = max_name_length(data);
  float const width = plot_breathing_room
    + legend_line_length
    + plot_breathing_room
    + char_count * plot_text_width
    + plot_breathing_room;
  float const height = 
    (plot_breathing_room + plot_text_height) * data.groups.size()
    + plot_breathing_room;
  return float_vec(width, height);
}

static void draw_legend_marker(
    coloring::canvas& canvas,
    plot_group const& group,
    float_vec const& left,
    float_vec const& right)
{
  for (auto& dataset : group.datasets) {
    if (dataset.style == plot_style::lines) {
      canvas.draw_line(
          left,
          right,
          group.color.value(),
          dataset.thickness.value_or(default_line_thickness));
    } else if (dataset.style == plot_style::points) {
      canvas.draw_point(
          (left + right) / 2,
          group.color.value(),
          dataset.thickness.value_or(default_point_thickness));
    } else {
      throw std::logic_error("unhandled style in legend drawing");
    }
  }
}

static void draw_legend_in_box(
    coloring::canvas& canvas,
    plot_data const& data,
    float_box const& box)
{
  canvas.draw_box(box, white);
  draw_box_outline(canvas, box, black, default_line_thickness);
  float_vec row_left = box.upper_left()
    + float_vec(plot_breathing_room,
        -plot_breathing_room - (plot_text_height / 2));
  for (auto& group : data.groups) {
    float_vec column_left = row_left;
    column_left = column_left
      + float_vec(legend_line_length, 0);
    draw_legend_marker(canvas, group, row_left, column_left);
    column_left = column_left
      + float_vec(plot_breathing_room, 0);
    canvas.draw_text(
        group.name,
        column_left,
        text_anchor::left,
        false,
        plot_text_width,
        black);
    row_left = row_left
      + float_vec(0, -plot_breathing_room - plot_text_height);
  }
}

static void draw_legend(
    coloring::canvas& canvas,
    plot_data const& data,
    float_box const& view_box)
{
  if (data.legend_location == legend_location::no_legend) return;
  float_vec const extents = compute_legend_extents(data);
  float_box box;
  if (data.legend_location == legend_location::upper_left) {
    box = float_box(
        view_box.upper_left()
        + float_vec(plot_breathing_room, -plot_breathing_room)
        + float_vec(0, -extents.y()),
        view_box.upper_left()
        + float_vec(plot_breathing_room, -plot_breathing_room)
        + float_vec(extents.x(), 0));
  } else if (data.legend_location == legend_location::upper_right) {
    box = float_box(
        view_box.upper_right()
        + float_vec(-plot_breathing_room, -plot_breathing_room)
        + float_vec(-extents.x(), -extents.y()),
        view_box.upper_right()
        + float_vec(-plot_breathing_room, -plot_breathing_room)
        + float_vec(0, 0));
  } else if (data.legend_location == legend_location::lower_right) {
    box = float_box(
        view_box.lower_right()
        + float_vec(-plot_breathing_room, plot_breathing_room)
        + float_vec(-extents.x(), 0),
        view_box.lower_right()
        + float_vec(-plot_breathing_room, plot_breathing_room)
        + float_vec(0, extents.y()));
  } else if (data.legend_location == legend_location::lower_left) {
    box = float_box(
        view_box.lower_left()
        + float_vec(plot_breathing_room, plot_breathing_room)
        + float_vec(0, 0),
        view_box.lower_left()
        + float_vec(plot_breathing_room, plot_breathing_room)
        + float_vec(extents.x(), extents.y()));
  } else {
    throw std::logic_error("unhandled legend_location option");
  }
  draw_legend_in_box(canvas, data, box);
}

inline float symlog(float x)
{
  if (x >= 0.0f) {
    return std::log(x + 1.0f);
  } else {
    return -std::log(-x + 1.0f);
  }
}

static void scale_axes(plot_data& data)
{
  if (data.xscale == axis_scale::log) {
    data.xlabel = "log(" + data.xlabel + ")";
  } else if (data.xscale == axis_scale::symlog) {
    data.xlabel = "symlog(" + data.xlabel + ")";
  }
  if (data.yscale == axis_scale::log) {
    data.ylabel = "log(" + data.ylabel + ")";
  } else if (data.yscale == axis_scale::symlog) {
    data.ylabel = "symlog(" + data.ylabel + ")";
  }
  for (auto& group : data.groups) {
    for (auto& dataset : group.datasets) {
      if (data.xscale == axis_scale::log) {
        for (auto& xvalue : dataset.x) xvalue = std::log(xvalue);
      } else if (data.xscale == axis_scale::symlog) {
        for (auto& xvalue : dataset.x) xvalue = symlog(xvalue);
      }
      if (data.yscale == axis_scale::log) {
        for (auto& yvalue : dataset.y) yvalue = std::log(yvalue);
      } else if (data.yscale == axis_scale::symlog) {
        for (auto& yvalue : dataset.y) yvalue = symlog(yvalue);
      }
    }
  }
}

void plot(
    std::filesystem::path const& pngpath,
    plot_data& data,
    unsigned width,
    unsigned height)
{
  int i = 0;
  for (auto& group : data.groups) {
    if (!group.color.has_value()) {
      group.color = byte_rgba(okabe_ito::colors[i % 7]);
      ++i;
    }
  }
  scale_axes(data);
  coloring::canvas canvas(width, height, white);
  float_box const data_box = compute_bounding_box(data);
  float_vec const data_extents = data_box.extents();
  float constexpr view_xmin = plot_breathing_room
    + plot_text_height
    + plot_breathing_room
    + tick_label_width
    + plot_breathing_room;
  float const view_xmax = width
    - plot_breathing_room
    - tick_label_width / 2;
  float constexpr view_ymin = plot_breathing_room
    + plot_text_height
    + plot_breathing_room
    + plot_text_height
    + plot_breathing_room;
  float const view_ymax = height
    - plot_breathing_room
    - plot_text_height
    - plot_breathing_room;
  float_box const view_box(
      float_vec(view_xmin, view_ymin),
      float_vec(view_xmax, view_ymax));
  float_vec const view_extents = view_box.extents();
  for (auto& group : data.groups) {
    draw_group(canvas, view_box, data_box, group);
  }
  canvas.draw_line(
      view_box.lower(),
      float_vec(view_xmax, view_ymin),
      black,
      default_line_thickness);
  canvas.draw_line(
      view_box.lower(),
      float_vec(view_xmin, view_ymax),
      black,
      default_line_thickness);
  char tick_buffer[tick_char_count + 1];
  std::vector<float> xticks;
  compute_ticks(data_box.lower().x(), data_box.upper().x(), xticks);
  for (float xtick : xticks) {
    int const print_result = std::snprintf(
        tick_buffer, sizeof(tick_buffer), "%.4g", double(xtick));
    if (print_result < 0) {
      throw std::runtime_error("formatting X ticks, snprintf returned negative code");
    }
    if (print_result >= sizeof(tick_buffer)) {
      throw std::runtime_error("formatting X ticks, snprintf ran out of space");
    }
    float_vec const data_point(xtick, data_box.lower().y());
    float_vec const tick_top = data_to_view(data_point, data_box, view_box);
    float_vec const tick_bottom = tick_top - float_vec(0, plot_breathing_room);
    canvas.draw_line(tick_top, tick_bottom, black, default_line_thickness);
    canvas.draw_text(tick_buffer, tick_bottom, text_anchor::top, false,
        plot_text_width, black);
  }
  std::vector<float> yticks;
  compute_ticks(data_box.lower().y(), data_box.upper().y(), yticks);
  for (float ytick : yticks) {
    int const print_result = std::snprintf(
        tick_buffer, sizeof(tick_buffer), "%.4g", double(ytick));
    if (print_result < 0) {
      throw std::runtime_error("formatting Y ticks, snprintf returned negative code");
    }
    if (print_result >= sizeof(tick_buffer)) {
      throw std::runtime_error("formatting Y ticks, snprintf ran out of space");
    }
    float_vec const data_point(data_box.lower().x(), ytick);
    float_vec const tick_right = data_to_view(data_point, data_box, view_box);
    float_vec const tick_left = tick_right - float_vec(plot_breathing_room, 0);
    canvas.draw_line(tick_left, tick_right, black, default_line_thickness);
    canvas.draw_text(tick_buffer, tick_left, text_anchor::right, false,
        plot_text_width, black);
  }
  float_vec const xname_anchor(
      (view_box.upper().x() + view_box.lower().x()) / 2,
      view_box.lower().y() - plot_breathing_room * 2 - plot_text_height);
  canvas.draw_text(data.xlabel, xname_anchor, text_anchor::top, false,
      plot_text_width, black);
  float_vec const yname_anchor(
      view_box.lower().x() - plot_breathing_room * 2 - tick_label_width,
      (view_box.upper().y() + view_box.lower().y()) / 2);
  canvas.draw_text(data.ylabel, yname_anchor, text_anchor::bottom, true,
      plot_text_width, black);
  float_vec const name_anchor(
      (view_box.upper().x() + view_box.lower().x()) / 2,
      view_box.upper().y() + plot_breathing_room);
  canvas.draw_text(data.title, name_anchor, text_anchor::bottom, false,
      plot_text_width, black);
  draw_legend(canvas, data, view_box);
  canvas.write(pngpath);
}

}
