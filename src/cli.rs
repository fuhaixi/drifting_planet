use crate::APP_NAME;


pub enum Command {
    New(std::fs::File),
    View(String),
    List,
    Init,
    None,
}



/// new <config file> --path <dir path>: create a planet with config and save data to directory
/// view <planet name> --path <dir path>: run window which rendring the University and set the planet as target
/// list --path <dir path> : list all planet 
/// init --path <dir path> : create world and save data to the directory
/// --path <dir path> : set the directory where the save data is located which is optional default is ~/.local/share/drifting_planet
/// 
/// save directory structure:
///   world.ron
///   planet1/
///      planet.ron
///      data.bin
///   planet2/
///    ...
pub fn parse_cmd_args(args: Vec<String>) -> (Command, std::path::PathBuf) {
    if args.len() < 2 {
        println!("no command specified");
        return (Command::None, std::path::PathBuf::new());
    }
    let cmd = args[1].as_str();


    
    let mut path = home::home_dir().unwrap().join(".local/share").join(APP_NAME);

    //create save directory if not exists
    if !path.exists() {
        std::fs::create_dir_all(&path).unwrap();
    }

    for i in 0..args.len() {
        if args[i] == "--path" {
            if let Some(path_str) = args.get(i+1) {
                path = std::path::PathBuf::from(path_str);
            }
            else {
                println!("no path specified, use default path: {}", path.to_str().unwrap());
            }
        }
    }


    match cmd {
        "new" => {
            if let Some(config_path) = args.get(2) {
                let config_file = std::fs::File::open(config_path).unwrap_or_else(|err| {
                    panic!("couldn't open {}: {}", config_path, err);
                });

                (Command::New(config_file), path)
            }
            else {
                println!("no config file specified");
                (Command::None, path)
            }
        }
        "view" => {
            if let Some(planet_name) = args.get(2) {
                (Command::View(planet_name.to_string()), path)
            }
            else {
                println!("no planet name specified");
                (Command::None, path)
            }
        }
        "list" => {
            (Command::List, path)
        }
        "init" => {
            (Command::Init, path)
        }
        _ => {
            println!("invalid command");
            (Command::None, path)
        }
    }

    

}
