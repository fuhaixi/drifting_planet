
mod cli;

pub const APP_NAME: &str = "DriftingPlanet";

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    
    let args: Vec<String> = std::env::args().collect();
    let (cmd, path) = cli::parse_cmd_args(args);
    match cmd {
        cli::Command::New(config_file) => {
            
            drifting_planet::new_planet(config_file, path).unwrap();
        }
        
        cli::Command::View(planet_name) => {
            pollster::block_on(drifting_planet::run(path) )
            
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
