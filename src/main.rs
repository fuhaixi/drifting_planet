use std::fs;
use std::path;
mod cli;

pub const APP_NAME: &str = "DriftingPlanet";

fn parse_cmd(){
    let args: Vec<String> = std::env::args().collect();
    let (cmd, path) = cli::parse_cmd_args(args);
    match cmd {
        cli::Command::New(config_file) => {
            
            drifting_planet::build_planet(config_file, path).unwrap();
        }
        
        cli::Command::View(planet_name) => {
            // pollster::block_on(drifting_planet::run_with_string(path, Some(planet_name) ))
            
        }
        
        cli::Command::Init => {
            drifting_planet::init_world(path).unwrap();
        }
        
        cli::Command::List => {
            
            drifting_planet::list_planets(path);
        }
        
        cli::Command::None => ()
    }
}

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let save_dir_path = path::PathBuf::from("./test/save");
    if !save_dir_path.exists(){
        fs::create_dir_all(&save_dir_path).unwrap();
    }

    pollster::block_on(drifting_planet::run_with_string(save_dir_path, Some("AA".to_string()) , "world_A" ));

   
}
