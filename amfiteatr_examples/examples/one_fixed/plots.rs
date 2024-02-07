use std::cmp::Ordering;
use std::path::Path;
use log::{debug, info};
use plotters::prelude::*;

pub struct Series{
    pub data: Vec<f32>,
    pub description: String,
    pub color: RGBColor,
}


pub fn plot_many_series(file: &Path, series: &[Series]) -> Result<(), Box<dyn std::error::Error>>{
    let root  = SVGBackend::new(&file, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut mins = Vec::with_capacity(series.len());
    for s in series{
        let min =  match s.data.iter().min_by(|a, b |{
            a.partial_cmp(b).unwrap_or(Ordering::Equal)
        }){
            None => 0.0,
            Some(n) if n < &0.0 => *n,
            Some(_) => 0.0f32
        };
        mins.push(min)
    }
    let global_min = match mins.iter().min_by(|a, b |{
            a.partial_cmp(b).unwrap_or(Ordering::Equal)
    }){
        None => 0.0,
        Some(n) if n < &0.0 => *n,
        Some(_) => 0.0f32
    };

    let mut maxes = Vec::with_capacity(series.len());
    for s in series{
        let max =  match s.data.iter().max_by(|a, b |{
            a.partial_cmp(b).unwrap_or(Ordering::Equal)
        }){
            None => 0.0,
            Some(n) if n > &0.0 => *n,
            Some(_) => 0.0f32
        };
        debug!("Maximal value in series: {} is {}", s.description, max);
        maxes.push(max)
    }
    let global_max = match maxes.iter().max_by(|a, b |{
            a.partial_cmp(b).unwrap_or(Ordering::Equal)
    }){
        None => 0.0,
        Some(n) if n > &0.0 => *n,
        Some(_) => 0.0f32
    };


    debug!("Plotting globals: min = {}; max = {}", global_min, global_max);

    /*
    let max = match series_0.data..iter().chain(series_1.data..iter()).max_by(|a, b |{
        a.partial_cmp(b).unwrap_or(Ordering::Equal)
    }){
        None => 0.0,
        Some(n) if n > &0.0 => *n,
        Some(_) => 0.0f32
    };

     */


    let mut chart = ChartBuilder::on(&root)
        .caption("payoffs", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..series[0].data.len() as f32, global_min..global_max)?;

    chart.configure_mesh().disable_mesh().draw()?;


    for s in series{
        chart
            .draw_series(LineSeries::new(
                (0..s.data.len()).map(|x| (x as f32, s.data[x])),
                &s.color,
            ))?
            .label(s.description.as_str())
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &s.color));

    }
    /*
    chart
        .draw_series(LineSeries::new(
            (0..payoffs_0.len()).map(|x| (x as f32, payoffs_0[x])),
            &RED,
        ))?
        .label("agent 0")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart

        .draw_series(LineSeries::new(
            (0..payoffs_1.len()).map(|x| (x as f32, payoffs_1[x])),
            &BLUE,
        ))?
        .label("agent 1")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

     */

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

