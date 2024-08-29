use image::{Luma, LumaA, Pixel, Primitive, Rgb, Rgba};

#[derive(Clone, Copy, Default)]
pub(crate) struct Lab<T>(pub [T; 3]);

pub(crate) trait PixelLabExt {
    type Subpixel;
    fn to_lab(&self) -> Lab<Self::Subpixel>;
}

impl PixelLabExt for Rgb<f32> {
    type Subpixel = f32;
    fn to_lab(&self) -> Lab<Self::Subpixel> {
        let Rgb(rgb) = self;
        Lab(rgb2lab(rgb))
    }
}

impl<T: Primitive> Pixel for Lab<T> {
    type Subpixel = T;

    const CHANNEL_COUNT: u8 = 3;

    fn channels(&self) -> &[Self::Subpixel] {
        &self.0
    }

    fn channels_mut(&mut self) -> &mut [Self::Subpixel] {
        &mut self.0
    }

    const COLOR_MODEL: &'static str = "LAB";

    fn channels4(
        &self,
    ) -> (
        Self::Subpixel,
        Self::Subpixel,
        Self::Subpixel,
        Self::Subpixel,
    ) {
        unimplemented!()
    }

    fn from_channels(
        a: Self::Subpixel,
        b: Self::Subpixel,
        c: Self::Subpixel,
        d: Self::Subpixel,
    ) -> Self {
        let _ = a;
        let _ = b;
        let _ = c;
        let _ = d;
        unimplemented!()
    }

    fn from_slice(slice: &[Self::Subpixel]) -> &Self {
        let _ = slice;
        todo!()
    }

    fn from_slice_mut(slice: &mut [Self::Subpixel]) -> &mut Self {
        let _ = slice;
        todo!()
    }

    fn to_rgb(&self) -> Rgb<Self::Subpixel> {
        todo!()
    }

    fn to_rgba(&self) -> Rgba<Self::Subpixel> {
        todo!()
    }

    fn to_luma(&self) -> Luma<Self::Subpixel> {
        todo!()
    }

    fn to_luma_alpha(&self) -> LumaA<Self::Subpixel> {
        todo!()
    }

    fn map<F>(&self, f: F) -> Self
    where
        F: FnMut(Self::Subpixel) -> Self::Subpixel,
    {
        let _ = f;
        todo!()
    }

    fn apply<F>(&mut self, f: F)
    where
        F: FnMut(Self::Subpixel) -> Self::Subpixel,
    {
        let _ = f;
        todo!()
    }

    fn map_with_alpha<F, G>(&self, f: F, g: G) -> Self
    where
        F: FnMut(Self::Subpixel) -> Self::Subpixel,
        G: FnMut(Self::Subpixel) -> Self::Subpixel,
    {
        let _ = f;
        let _ = g;
        todo!()
    }

    fn apply_with_alpha<F, G>(&mut self, f: F, g: G)
    where
        F: FnMut(Self::Subpixel) -> Self::Subpixel,
        G: FnMut(Self::Subpixel) -> Self::Subpixel,
    {
        let _ = f;
        let _ = g;
        todo!()
    }

    fn map2<F>(&self, other: &Self, f: F) -> Self
    where
        F: FnMut(Self::Subpixel, Self::Subpixel) -> Self::Subpixel,
    {
        let _ = other;
        let _ = f;
        todo!()
    }

    fn apply2<F>(&mut self, other: &Self, f: F)
    where
        F: FnMut(Self::Subpixel, Self::Subpixel) -> Self::Subpixel,
    {
        let _ = other;
        let _ = f;
        todo!()
    }

    fn invert(&mut self) {
        todo!()
    }

    fn blend(&mut self, other: &Self) {
        let _ = other;
        todo!()
    }

    fn map_without_alpha<F>(&self, f: F) -> Self
    where
        F: FnMut(Self::Subpixel) -> Self::Subpixel,
    {
        let mut this = *self;
        this.apply_with_alpha(f, |x| x);
        this
    }

    fn apply_without_alpha<F>(&mut self, f: F)
    where
        F: FnMut(Self::Subpixel) -> Self::Subpixel,
    {
        self.apply_with_alpha(f, |x| x);
    }
}

/// https://en.wikipedia.org/wiki/CIELAB_color_space
/// lを２倍に
fn rgb2lab(rgb: &[f32; 3]) -> [f32; 3] {
    let [mut r, mut g, mut b] = rgb.map(|c| c / 255.0);
    r = if r > 0.04045 {
        f32::powf((r + 0.055) / 1.055, 2.4)
    } else {
        r / 12.92
    };
    g = if g > 0.04045 {
        f32::powf((g + 0.055) / 1.055, 2.4)
    } else {
        g / 12.92
    };
    b = if b > 0.04045 {
        f32::powf((b + 0.055) / 1.055, 2.4)
    } else {
        b / 12.92
    };
    let mut x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
    let mut y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000;
    let mut z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;
    x = if x > 0.008856 {
        f32::powf(x, 1.0 / 3.0)
    } else {
        (7.787 * x) + 16.0 / 116.0
    };
    y = if y > 0.008856 {
        f32::powf(y, 1.0 / 3.0)
    } else {
        (7.787 * y) + 16.0 / 116.0
    };
    z = if z > 0.008856 {
        f32::powf(z, 1.0 / 3.0)
    } else {
        (7.787 * z) + 16.0 / 116.0
    };
    let l = (116.0 * y) - 16.0;
    let a = 500.0 * (x - y);
    let b = 200.0 * (y - z);
    [2.0 * l, a, b]
}

