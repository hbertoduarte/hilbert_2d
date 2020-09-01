use hilbert_2d::{h2xy_discrete, Variant};
use palette::{Hsv, Pixel, Srgb};
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{LineStyle, PointStyle};
use plotlib::view::ContinuousView;

/// Plot the discrete approximations of all Hilbert curve variants, for orders 2, 3, and 4
fn main() {
    // Which orders and variants to plot
    let orders = vec![2, 3, 4];
    let variants = vec![
        Variant::Hilbert,
        Variant::Moore,
        Variant::Liu1,
        Variant::Liu2,
        Variant::Liu3,
        Variant::Liu4,
    ];

    // Plot every `order x variant` combination
    let mut lines: Vec<Plot> = Vec::new();
    for oi in 0..orders.len() {
        for vi in 0..variants.len() {
            let order = orders[oi];
            let variant = variants[vi];

            // The side length of a square covered by a Hilbert curve approximation is `2^order`
            let side_length = 2f64.powi(order as i32);

            // The number of discrete cells in a Hilbert curve approximation is `2^(2 * order)`
            let num_points = 2usize.pow(2 * order as u32);

            let mut points = Vec::new();
            for h in 0..num_points {
                // Get all points for this curve
                let (p_x, p_y) = h2xy_discrete(h, order, variant);

                // Position the point in the center of the discrete cell it represents
                points.push((
                    vi as f64 + ((p_x as f64) + 0.5) / side_length,
                    oi as f64 + ((p_y as f64) + 0.5) / side_length,
                ));
            }

            // Plot a colored line segment between each point for this curve
            for pi in 1..points.len() {
                let color = get_color(pi - 1, points.len());
                let width = 2.0 * (orders.len() - oi) as f32;
                lines.push(
                    Plot::new(vec![points[pi - 1], points[pi]])
                        .line_style(LineStyle::new().width(width).colour(color.as_str()))
                        .point_style(PointStyle::new().size(0.5 * width).colour(color.as_str())),
                );
            }
        }
    }

    // Create a view, and add to it all line plots
    let view = lines.into_iter().fold(
        ContinuousView::new()
            .x_range(0.0, variants.len() as f64)
            .y_range(0.0, orders.len() as f64),
        |v, p| v.add(p),
    );
    // Render plot to svg
    Page::single(&view)
        .dimensions(72 + (variants.len() * 260) as u32, 36 + (orders.len() * 260) as u32)
        .save("all_discrete.svg")
        .unwrap();
}

// Assign a color to a value between 0 and `max`
fn get_color(cur: usize, max: usize) -> String {
    let hue = (cur * 360) as f64 / (max as f64);
    let a = Hsv::new(hue, 1.0, 1.0);
    let b: [u8; 3] = Srgb::into_raw(Srgb::from(a).into_format());
    format!("#{:02x}{:02x}{:02x}", b[0], b[1], b[2])
}
