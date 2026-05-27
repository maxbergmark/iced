use std::marker::PhantomData;
use std::ops::RangeBounds;

#[cfg(not(target_arch = "wasm32"))]
use std::num::NonZeroU64;

pub const MAX_WRITE_SIZE: usize = 100 * 1024;

/// A buffer upload abstraction that bundles a `wgpu::util::StagingBelt`
/// with a `wgpu::Queue`.
///
/// On native targets, [`Belt::write_buffer`] uses the staging belt's
/// mapped staging buffer + `copyBufferToBuffer` path, which is the most
/// efficient route on Vulkan / Metal / DX12.
///
/// On `wasm32` targets, [`Belt::write_buffer`] bypasses the staging belt
/// entirely and uses [`wgpu::Queue::write_buffer`] directly. This works
/// around a WebKit WebGPU bug where the `mappedAtCreation` staging buffer +
/// `copyBufferToBuffer` pattern triggers a silent `Validation failure.` at
/// `Queue::submit` (see <https://github.com/iced-rs/iced/issues/...>).
///
/// For advanced upload patterns (e.g. `copy_buffer_to_texture` from a
/// staged buffer), [`Belt::inner_mut`] exposes the underlying
/// `wgpu::util::StagingBelt`. The same WebKit bug applies to that path on
/// `wasm32`; callers that need wasm support should switch to
/// [`wgpu::Queue::write_texture`] instead.
#[derive(Debug)]
pub struct Belt {
    inner: wgpu::util::StagingBelt,
    // Only used on wasm32 to bypass the staging belt; see `write_buffer`.
    #[cfg(target_arch = "wasm32")]
    queue: wgpu::Queue,
}

impl Belt {
    pub fn new(queue: &wgpu::Queue, chunk_size: u64) -> Self {
        let _ = queue;
        Self {
            inner: wgpu::util::StagingBelt::new(chunk_size),
            #[cfg(target_arch = "wasm32")]
            queue: queue.clone(),
        }
    }

    /// Writes `bytes` into `buffer` at `offset`.
    ///
    /// See [`Belt`] for the platform-specific implementation strategy.
    pub fn write_buffer(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        buffer: &wgpu::Buffer,
        offset: u64,
        bytes: &[u8],
        device: &wgpu::Device,
    ) {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (encoder, device);
            self.queue.write_buffer(buffer, offset, bytes);
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(size) = NonZeroU64::new(bytes.len() as u64) else {
                return;
            };
            self.inner
                .write_buffer(encoder, buffer, offset, size, device)
                .copy_from_slice(bytes);
        }
    }

    /// Exposes the underlying [`wgpu::util::StagingBelt`] for advanced uses
    /// such as [`wgpu::util::StagingBelt::allocate`] followed by a
    /// `copy_buffer_to_texture` (e.g. atlas uploads).
    ///
    /// Note: on `wasm32` this path is still affected by the WebKit WebGPU
    /// bug described in [`Belt`]'s documentation.
    #[cfg_attr(not(any(feature = "image", feature = "svg")), allow(dead_code))]
    pub fn inner_mut(&mut self) -> &mut wgpu::util::StagingBelt {
        &mut self.inner
    }

    pub fn finish(&mut self) {
        self.inner.finish();
    }

    pub fn recall(&mut self) {
        self.inner.recall();
    }
}

#[derive(Debug)]
pub struct Buffer<T> {
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
    pub(crate) raw: wgpu::Buffer,
    type_: PhantomData<T>,
}

impl<T: bytemuck::Pod> Buffer<T> {
    pub fn new(
        device: &wgpu::Device,
        label: &'static str,
        amount: usize,
        usage: wgpu::BufferUsages,
    ) -> Self {
        let size = next_copy_size::<T>(amount);

        let raw = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });

        Self {
            label,
            size,
            usage,
            raw,
            type_: PhantomData,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, new_count: usize) -> bool {
        let new_size = next_copy_size::<T>(new_count);

        if self.size < new_size {
            self.raw = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: new_size,
                usage: self.usage,
                mapped_at_creation: false,
            });

            self.size = new_size;

            true
        } else {
            false
        }
    }

    /// Returns the size of the written bytes.
    pub fn write(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        belt: &mut Belt,
        offset: usize,
        contents: &[T],
    ) -> usize {
        let bytes: &[u8] = bytemuck::cast_slice(contents);

        // On wasm32, `Belt::write_buffer` uses `queue.write_buffer` which
        // has no per-call size limit and handles its own chunking, so we
        // can skip the manual chunking loop.
        #[cfg(target_arch = "wasm32")]
        {
            belt.write_buffer(encoder, &self.raw, offset as u64, bytes, device);
            return bytes.len();
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut bytes_written = 0;

            // Split write into multiple chunks if necessary
            while bytes_written + MAX_WRITE_SIZE < bytes.len() {
                belt.write_buffer(
                    encoder,
                    &self.raw,
                    (offset + bytes_written) as u64,
                    &bytes[bytes_written..bytes_written + MAX_WRITE_SIZE],
                    device,
                );

                bytes_written += MAX_WRITE_SIZE;
            }

            // Write the remaining bytes (always non-empty thanks to the
            // strict `<` above).
            belt.write_buffer(
                encoder,
                &self.raw,
                (offset + bytes_written) as u64,
                &bytes[bytes_written..],
                device,
            );

            bytes.len()
        }
    }

    pub fn slice(
        &self,
        bounds: impl RangeBounds<wgpu::BufferAddress>,
    ) -> wgpu::BufferSlice<'_> {
        self.raw.slice(bounds)
    }

    pub fn range(&self, start: usize, end: usize) -> wgpu::BufferSlice<'_> {
        self.slice(
            start as u64 * std::mem::size_of::<T>() as u64
                ..end as u64 * std::mem::size_of::<T>() as u64,
        )
    }
}

fn next_copy_size<T>(amount: usize) -> u64 {
    let align_mask = wgpu::COPY_BUFFER_ALIGNMENT - 1;

    (((std::mem::size_of::<T>() * amount).next_power_of_two() as u64
        + align_mask)
        & !align_mask)
        .max(wgpu::COPY_BUFFER_ALIGNMENT)
}
