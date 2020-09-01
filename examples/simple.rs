use hilbert_2d::{h2xy_discrete, Variant};
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::LineStyle;
use plotlib::view::ContinuousView;

/// Calculates and plots the Hilbert curve approximations of order 1, 2 and 3
fn main() {
    let variant = Variant::Hilbert;

    // Calculate the 4 coordinates for a Hilbert curve approximation of order 1 
    let mut order1 = Vec::new();
    for h in 0..4 {
        order1.push(h2xy_discrete(h, 1, variant));
    }

    // Calculate the 16 coordinates for a Hilbert curve approximation of order 2  
    let mut order2 = Vec::new();
    for h in 0..16 {
        order2.push(h2xy_discrete(h, 2, variant));
    }

    // Calculate the 64 coordinates for a Hilbert curve approximation of order 3  
    let mut order3 = Vec::new();
    for h in 0..64 {
        order3.push(h2xy_discrete(h, 3, variant));
    }

    // Normalize all coordinates to the unit square
    let line1 = order1
        .into_iter()
        .map(|(x, y)| (((x as f64) + 0.5) / 2.0, ((y as f64) + 0.5) / 2.0))
        .collect();
    let line2 = order2
        .into_iter()
        .map(|(x, y)| (((x as f64) + 0.5) / 4.0, ((y as f64) + 0.5) / 4.0))
        .collect();
    let line3 = order3
        .into_iter()
        .map(|(x, y)| (((x as f64) + 0.5) / 8.0, ((y as f64) + 0.5) / 8.0))
        .collect();

    // Plot lines and render to svg
    let plot1 = Plot::new(line1).line_style(LineStyle::new().colour("#f00"));
    let plot2 = Plot::new(line2).line_style(LineStyle::new().colour("#0f0"));
    let plot3 = Plot::new(line3).line_style(LineStyle::new().colour("#00f"));
    let view = ContinuousView::new()
        .add(plot1)
        .add(plot2)
        .add(plot3)
        .x_range(0.0, 1.0)
        .y_range(0.0, 1.0);
    Page::single(&view)
        .dimensions(700 + 72, 700 + 36)
        .save("simple.svg")
        .unwrap();
}
