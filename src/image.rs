use std::path::Path;
use image::{io::Reader, imageops};

pub const RESIZE: usize = 416;

pub struct Image {
    data: Vec<u8>,
    img_len: usize,
    // height_size: usize,
    // width_size: usize,
}

impl Image {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let img = Reader::open(path)?.decode()?.into_luma8();
        let resized_img = imageops::resize(&img, RESIZE as u32, RESIZE as u32, imageops::FilterType::Lanczos3);
        let d: Vec<u8> = resized_img.into_raw();
        let len = d.len();
        // println!("将图像转换成长度为{}的 Vector", len);
        // let hs = height as usize;
        // let ws = width as usize;
        Ok(
                Image {
                data: d,
                img_len: len
                // height_size: hs,
                // width_size: ws,
            }
        )
    }

    pub fn extend<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let img = Reader::open(path)?.decode()?.into_luma8();
        let resized_img = imageops::resize(&img, RESIZE as u32, RESIZE as u32, imageops::FilterType::Lanczos3);
        let ri = resized_img.into_raw();
        self.data.extend(&ri);
        // println!("将图像转换成长度为{}的 Vector 并接续后长度为{}", ri.len(), self.data.len());
        Ok(())
    }

    pub fn get_data(&self) -> Vec<u8> {
        self.data.clone()
    }

    pub fn len(&self) -> usize {
        self.img_len
    }
}
