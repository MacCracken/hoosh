//! hoosh CLI — inference gateway server and management tool.
//!
//! Usage:
//!   hoosh serve                         # Start the gateway server (port 8088)
//!   hoosh serve --port 9000             # Custom port
//!   hoosh models                        # List available models across providers
//!   hoosh infer --model llama3 "hello"  # One-shot inference
//!   hoosh health                        # Check provider health
//!   hoosh info                          # System info (providers, hardware)
//!   hoosh --version

use clap::{Parser, Subcommand};
use hoosh::client::HooshClient;
use hoosh::server::ServerConfig;

#[derive(Parser)]
#[command(name = "hoosh", version, about = "AI inference gateway")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the inference gateway server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "8088")]
        port: u16,
        /// Bind address
        #[arg(long, default_value = "127.0.0.1")]
        bind: String,
    },
    /// List available models across all providers
    Models {
        /// hoosh server URL
        #[arg(long, default_value = "http://127.0.0.1:8088")]
        server: String,
    },
    /// Run a one-shot inference
    Infer {
        /// Model identifier
        #[arg(short, long)]
        model: String,
        /// Prompt text
        prompt: String,
        /// Stream output token by token
        #[arg(long)]
        stream: bool,
        /// hoosh server URL
        #[arg(long, default_value = "http://127.0.0.1:8088")]
        server: String,
    },
    /// Check health of all configured providers
    Health {
        /// hoosh server URL
        #[arg(long, default_value = "http://127.0.0.1:8088")]
        server: String,
    },
    /// Show system info (providers, hardware, config)
    Info,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { port, bind } => {
            let config = ServerConfig {
                bind,
                port,
                ..Default::default()
            };
            hoosh::server::run(config).await?;
        }
        Commands::Models { server } => {
            let client = HooshClient::new(&server);
            match client.list_models().await {
                Ok(models) => {
                    if models.is_empty() {
                        println!("No models available.");
                    } else {
                        for m in &models {
                            println!(
                                "  {} (provider: {}, available: {})",
                                m.id, m.provider, m.available
                            );
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to connect to hoosh server at {}: {}", server, e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Infer {
            model,
            prompt,
            stream: _,
            server,
        } => {
            let client = HooshClient::new(&server);
            let req = hoosh::InferenceRequest {
                model,
                prompt,
                ..Default::default()
            };
            match client.infer(&req).await {
                Ok(resp) => println!("{}", resp.text),
                Err(e) => {
                    eprintln!("Inference failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Health { server } => {
            let client = HooshClient::new(&server);
            match client.health().await {
                Ok(true) => println!("hoosh server at {} is healthy", server),
                Ok(false) => {
                    println!("hoosh server at {} is unhealthy", server);
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("Health check failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Info => {
            println!("hoosh v{}", env!("CARGO_PKG_VERSION"));
            println!();

            #[cfg(feature = "hwaccel")]
            {
                let registry = ai_hwaccel::AcceleratorRegistry::detect();
                println!("Hardware:");
                for p in registry.all_profiles() {
                    let mem_gb = p.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                    println!("  {} ({:.1} GB)", p.accelerator, mem_gb);
                }
                println!();
            }

            #[cfg(feature = "whisper")]
            println!("Speech-to-text: whisper.cpp (enabled)");
            #[cfg(not(feature = "whisper"))]
            println!("Speech-to-text: disabled (enable 'whisper' feature)");
        }
    }

    Ok(())
}
