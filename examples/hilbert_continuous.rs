use hilbert_2d::{h2xy_continuous_f64, Variant};
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{LineStyle, PointStyle};
use plotlib::view::ContinuousView;

/// Plots the lines for slices of the limiting Hilbert curve, for different step lenghts
fn main() {
    let variant = Variant::Hilbert;
    let steps = vec![0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625];
    let colors = vec!["#f00", "#ee0", "#0f0", "#0ff", "#00f", "#f0f"];

    let mut lines: Vec<Plot> = Vec::new();
    for i in 0..steps.len() {
        let step = steps[i];

        // Get the points between each slice of size `step` of the Hilbert curve
        let mut points = Vec::new();
        let mut acc = 0.0;
        while acc <= 1.0 {
            let (x, y) = h2xy_continuous_f64(acc, variant);
            acc += step;

            // Shift the points to the side
            points.push((x + 1.05 * i as f64, y));
        }

        // Plot a line between each sline
        lines.push(
            Plot::new(points)
                .line_style(LineStyle::new().width(1.5).colour(colors[i]))
                .point_style(PointStyle::new().size(3.0).colour(colors[i])),
        );
    }

    // Create a view, and add to it all line plots
    let view = lines.into_iter().fold(
        ContinuousView::new()
            .x_range(-0.05, 0.25 + steps.len() as f64)
            .y_range(-0.05, 1.05),
        |v, p| v.add(p),
    );
    // Render plot to svg
    Page::single(&view)
        .dimensions(72 + (steps.len() * 225) as u32, 36 + 250)
        .save("hilbert_continuous.svg")
        .unwrap();
}
