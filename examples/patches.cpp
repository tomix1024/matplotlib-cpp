#define _USE_MATH_DEFINES
#include <cmath>
#include "../matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main() 
{
	auto patch = plt::patches::Ellipse(10, 10, 4, 2, 45);
    auto patch2 = plt::patches::Rectangle(5, 2, 3, 3, {{"facecolor", "red"}});

    patch2.set_alpha(.5);

    patch.set_label("Ellipse");
    patch2.set_label("Rectangle");

    plt::add_patch(patch);
    plt::add_patch(patch2);

	// Add graph title
	plt::title("Sample figure");

	plt::xlim(0, 20);
	plt::ylim(0, 20);

    plt::legend();

	// save figure
	//plt::save("./patches.png");
	plt::show();
}
