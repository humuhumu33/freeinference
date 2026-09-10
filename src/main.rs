#![forbid(unsafe_code)]

#[tokio::main]
async fn main() {
    freeinference::cli::run(freeinference::modules::webgpu::select_engine).await;
}
