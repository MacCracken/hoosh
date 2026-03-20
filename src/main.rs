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
    Models,
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
    },
    /// Check health of all configured providers
    Health,
    /// Show system info (providers, hardware, config)
    Info,
}

fn main() {
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
            println!(
                "hoosh v{} — AI inference gateway",
                env!("CARGO_PKG_VERSION")
            );
            println!("Listening on {}:{}", bind, port);
            println!(
                "OpenAI-compatible API: http://{}:{}/v1/chat/completions",
                bind, port
            );
            println!("(server not yet implemented — scaffold only)");
        }
        Commands::Models => {
            println!("(model listing not yet implemented — scaffold only)");
        }
        Commands::Infer {
            model,
            prompt,
            stream,
        } => {
            println!(
                "Infer: model={}, stream={}, prompt=\"{}\"",
                model,
                stream,
                if prompt.len() > 60 {
                    format!("{}...", &prompt[..60])
                } else {
                    prompt
                }
            );
            println!("(inference not yet implemented — scaffold only)");
        }
        Commands::Health => {
            println!("(health check not yet implemented — scaffold only)");
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

            println!("Configured providers:");
            println!("  (provider configuration not yet loaded — scaffold only)");

            #[cfg(feature = "whisper")]
            println!("\nSpeech-to-text: whisper.cpp (enabled)");
            #[cfg(not(feature = "whisper"))]
            println!("\nSpeech-to-text: disabled (enable 'whisper' feature)");
        }
    }
}
