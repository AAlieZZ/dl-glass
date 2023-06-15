use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use serde_xml_rs::from_reader;

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
            None => names.push(4),
            Some(o) => for name in o {
                names.push(name.name);
            }
        }
        names
    }

    // pub fn get_name(&self) -> Vec<u8> {
    //     let mut names = Vec::new();
    //     for name in &self.object {
    //         names.push(name.name);
    //     }
    //     names
    // }

//     pub fn get_xmin(&self) -> Vec<u32> {
//         let mut xmins = Vec::new();
//         for xmin in &self.object {
//             xmins.push(xmin.bndbox.xmin);
//         }
//         xmins
//     }

//     pub fn get_ymin(&self) -> Vec<u32> {
//         let mut ymins = Vec::new();
//         for ymin in &self.object {
//             ymins.push(ymin.bndbox.ymin);
//         }
//         ymins
//     }

//     pub fn get_xmax(&self) -> Vec<u32> {
//         let mut xmaxs = Vec::new();
//         for xmax in &self.object {
//             xmaxs.push(xmax.bndbox.xmin);
//         }
//         xmaxs
//     }

//     pub fn get_ymax(&self) -> Vec<u32> {
//         let mut ymaxs = Vec::new();
//         for ymax in &self.object {
//             ymaxs.push(ymax.bndbox.ymin);
//         }
//         ymaxs
//     }
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
    xmin: u32,
    ymin: u32,
    xmax: u32,
    ymax: u32,
}