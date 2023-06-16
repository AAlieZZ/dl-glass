use std::path::Path;
use image::{io::Reader, imageops, imageops::crop};

pub const RESIZE: usize = 64;

pub struct Image {
    data: Vec<u8>,
    img_len: usize,
    // height_size: usize,
    // width_size: usize,
}

impl Image {
    pub fn new<P: AsRef<Path>>(path: P, mut xmin: i32, mut ymin: i32, mut xmax: i32, mut ymax: i32) -> Result<Self, Box<dyn std::error::Error>> {
        let mut img = Reader::open(path)?.decode()?.into_luma8();
        let (width, height) = img.dimensions();
        if xmax - xmin < RESIZE as i32 && ymax - ymin < RESIZE as i32 {
            xmin -= RESIZE as i32 / 2;
            ymin -= RESIZE as i32 / 2;
            xmax = xmin + RESIZE as i32;
            ymax = ymin + RESIZE as i32;
            if xmin < 0 {
                xmin = 0; xmax = RESIZE as i32;
            }
            if ymin < 0 {
                ymin = 0; ymax = RESIZE as i32;
            }
            if xmax > width as i32 {
                xmax = width as i32; xmin = width as i32 - RESIZE as i32;
            }
            if ymax > height as i32 {
                ymax = height as i32; ymin = height as i32 - RESIZE as i32;
            }
        }
        let resized_img;
        match xmin + ymin + xmax + ymax {
            0 => resized_img = imageops::resize(&img, RESIZE as u32, RESIZE as u32, imageops::FilterType::Lanczos3),
            _ => resized_img = imageops::resize(&crop(&mut img, xmin as u32, ymin as u32, xmax as u32, ymax as u32), RESIZE as u32, RESIZE as u32, imageops::FilterType::Lanczos3),
        }
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

    pub fn extend<P: AsRef<Path>>(&mut self, path: P,  mut xmin: i32, mut ymin: i32, mut xmax: i32, mut ymax: i32) -> Result<(), Box<dyn std::error::Error>> {
        let mut img = Reader::open(path)?.decode()?.into_luma8();
        let (width, height) = img.dimensions();
        if xmax - xmin < RESIZE as i32 && ymax - ymin < RESIZE as i32 {
            xmin -= RESIZE as i32 / 2;
            ymin -= RESIZE as i32 / 2;
            xmax = xmin + RESIZE as i32;
            ymax = ymin + RESIZE as i32;
            if xmin < 0 {
                xmin = 0; xmax = RESIZE as i32;
            }
            if ymin < 0 {
                ymin = 0; ymax = RESIZE as i32;
            }
            if xmax > width as i32 {
                xmax = width as i32; xmin = width as i32 - RESIZE as i32;
            }
            if ymax > height as i32 {
                ymax = height as i32; ymin = height as i32 - RESIZE as i32;
            }
        }
        let resized_img;
        match xmin + ymin + xmax + ymax {
            0 => resized_img = imageops::resize(&img, RESIZE as u32, RESIZE as u32, imageops::FilterType::Lanczos3),
            _ => resized_img = imageops::resize(&crop(&mut img, xmin as u32, ymin as u32, xmax as u32, ymax as u32), RESIZE as u32, RESIZE as u32, imageops::FilterType::Lanczos3),
        }
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