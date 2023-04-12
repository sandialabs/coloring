#include "coloring.hpp"

int main()
{
  std::vector<float> x = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float> me_y = {0.0f, 1.0f, 1.0f, 2.0f};
  std::vector<float> dog_y = {0.0f, 0.8f, 1.6f, 2.0f};
  coloring::plot_dataset me_dataset;
  me_dataset.x = x;
  me_dataset.y = me_y;
  coloring::plot_group me_group;
  me_group.datasets = {me_dataset};
  me_group.name = "me";
  coloring::plot_dataset dog_dataset;
  dog_dataset.x = {x};
  dog_dataset.y = {dog_y};
  coloring::plot_group dog_group;
  dog_group.name = "my dog";
  dog_group.datasets = {dog_dataset};
  coloring::plot_data data;
  data.groups = {me_group, dog_group};
  data.xlabel = "Time (hr)";
  data.ylabel = "Distance (km)";
  data.title = "Hiking Trip";
  data.legend_location = coloring::legend_location::upper_left;
  coloring::plot("example.png", data);
}
