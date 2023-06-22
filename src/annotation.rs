use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use serde_xml_rs::from_reader;
use crate::image::RESIZE;

#[derive(Serialize, Deserialize)]
pub struct Annotation {
    path: String,
    object: Option<Vec<Object>>,
}

impl Annotation {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(
            from_reader(
                BufReader::new(
                    File::open(path)?
                )
            )?
        )
    }

    pub fn get_path(&self) -> &str {
        &self.path
    }

    pub fn get_name(&self) -> Vec<u8> {
        let mut names = Vec::new();
        match &self.object {
            None => names.push(3),
            Some(o) => for name in o {
                names.push(name.name);
            }
        }
        names
    }

    pub fn get_xmin(&self) -> Vec<f32> {
        let mut xmins = Vec::new();
        match &self.object {
            None => xmins.push(0.0),
            Some(o) => for xmin in o {
                xmins.push(xmin.bndbox.xmin / RESIZE as f32);
            }
        }
        xmins
    }

    pub fn get_ymin(&self) -> Vec<f32> {
        let mut ymins = Vec::new();
        match &self.object {
            None => ymins.push(0.0),
            Some(o) => for ymin in o {
                ymins.push(ymin.bndbox.ymin / RESIZE as f32);
            }
        }
        ymins
    }

    pub fn get_xmax(&self) -> Vec<f32> {
        let mut xmaxs = Vec::new();
        match &self.object {
            None => xmaxs.push(0.0),
            Some(o) => for xmax in o {
                xmaxs.push(xmax.bndbox.xmax / RESIZE as f32);
            }
        }
        xmaxs
    }

    pub fn get_ymax(&self) -> Vec<f32> {
        let mut ymaxs = Vec::new();
        match &self.object {
            None => ymaxs.push(0.0),
            Some(o) => for ymax in o {
                ymaxs.push(ymax.bndbox.ymax / RESIZE as f32);
            }
        }
        ymaxs
    }
}

#[derive(Serialize, Deserialize)]
struct Object {
    name: u8,
    pose: String,
    truncated: u32,
    difficult: u32,
    bndbox: Bndbox,
}

#[derive(Serialize, Deserialize)]
struct Bndbox {
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
}