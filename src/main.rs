use std::fs;
use std::path;


pub const APP_NAME: &str = "DriftingPlanet";

use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    ///build terrain data
    Build(BuildArgs),
    
    ///view planet
    View(ViewArgs),

    ///init world
    Init(InitArgs),
}

#[derive(Args)]
struct InitArgs {
    #[arg(short, long, value_name = "WorldName")]
    name: Option<String>,

    #[arg(short, long, value_name = "File")]
    dir: Option<std::path::PathBuf>,
}

#[derive(Args)]
struct ViewArgs {
    #[arg(short, long, value_name = "File")]
    dir: Option<std::path::PathBuf>,
    
    #[arg(short, long, value_name = "World")]
    world: Option<String>,

    #[arg(short, long, value_name = "Planet")]
    planet: Option<String>,
}

#[derive(Args)]
struct BuildArgs {
    ///config file
    #[arg(short, long, value_name = "File")]
    config: std::path::PathBuf,

    #[arg(short, long)]
    save_dir: Option<std::path::PathBuf>,
}

fn main() {

    let cli = Cli::parse();

    match &cli.command {
        Commands::Build(args) => {
            let config_path = args.config.clone();
            let config_file = std::fs::File::open(config_path.clone()).unwrap_or_else(|err| {
                panic!("couldn't open {}: {}", config_path.to_str().unwrap(), err);
            });
            let save_dir_path = args.save_dir.clone().unwrap_or(path::PathBuf::from("./World_A"));
            if !save_dir_path.exists(){
                fs::create_dir_all(&save_dir_path).unwrap();
            }

            drifting_planet::build_planet(config_file, save_dir_path).unwrap();
        }

        Commands::View(args) =>{
            let save_dir_path = args.dir.clone().unwrap_or(path::PathBuf::from("./"));
            let world_name = args.world.clone().unwrap_or("World_A".to_string());
            let planet_name = args.planet.clone().unwrap_or("AA".to_string());
            pollster::block_on(drifting_planet::run_with_string(save_dir_path.clone(), Some(planet_name.clone()) , &world_name ));
        }

        Commands::Init(args) => {
            let save_dir_path = args.dir.clone().unwrap_or(path::PathBuf::from("./"));
            if !save_dir_path.exists(){
                fs::create_dir_all(&save_dir_path).unwrap();
            }
            let world_name = args.name.clone().unwrap_or("World_A".to_string());
            drifting_planet::init_world(save_dir_path, world_name).unwrap();
        }
    }
 
   
}
