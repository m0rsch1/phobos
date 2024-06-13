#!python3


def can_be_used():
    return True


def cant_be_used_msg():
    return ""


INFO = 'Converts the given input robot file to SDF/URDF/PDF/THUMBNAIL/SMURF/SMURFA/SMURFS.'


def main(args):
    import argparse
    import os.path as path
    from ..defs import load_json
    from ..core import Robot, World
    from ..utils import misc, resources
    from ..commandline_logging import setup_logger_level, BASE_LOG_LEVEL

    parser = argparse.ArgumentParser(description=INFO, prog="phobos " + path.basename(__file__)[:-3])
    parser.add_argument('input', type=str, help='Path to the urdf or smurf file')
    parser.add_argument('output', type=str, help='Writes the result to the given file. '
                                                 'The output is determined by the file ending, '
                                                 'if the output has no ending like sdf, urdf or pdf and '
                                                 'is a (non-existing) directory SMURF will be exported.',
                        action="store", default=None)
    parser.add_argument('-c', '--copy-meshes', help='Copies the meshes', action='store_true', default=False)
    parser.add_argument('-m', '--mesh-type', choices=["stl", "obj", "mars_obj", "dae", "input_type"], help='If -c set the mesh type', default="input_type")
    parser.add_argument('-a', '--sdf-assemble', help='When converting a scene to sdf set this flag to get one assembled model', action='store_true', default=False)
    parser.add_argument('-e', '--export_config', type=str, help='Path to the a json/yaml file that stores the export config', default=None)
    parser.add_argument("--loglevel", help="The log level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default=BASE_LOG_LEVEL)
    args = parser.parse_args(args)
    log = setup_logger_level(log_level=args.loglevel)

    if args.input == args.output:
        return 0
    if args.output.endswith("/"):
        args.output = args.output[:-1]

    world = None
    robot = None
    if args.input.rsplit(".", 1)[-1] in ["smurfa", "smurfs"]:
        world = World(inputfile=args.input)
        if world.is_empty():
            log.error("Given file is empty!")
            return 1
        if args.output.lower() not in ["smurfa", "smurfs"] and world.has_one_root():
            robot = world.assemble()
    else:
        robot = Robot(inputfile=args.input)

    if args.copy_meshes and robot and not args.export_config:
        robot.export_meshes(path.join(path.dirname(args.output), "meshes"), format=args.mesh_type)
    if args.export_config:
        assert path.isfile(args.export_config)
        with open(args.export_config, "r") as f:
            export_config = load_json(f.read())
        robot.export(args.output, export_config=export_config["export_config"], with_meshes=args.copy_meshes, no_smurf=["no_smurf"])
    elif args.output.lower().endswith("sdf"):
        if args.sdf_assemble:  # world
            log.info("Converting to SDF model")
            robot.export_sdf(outputfile=args.output, mesh_format=args.mesh_type)
        else:  # world
            log.info("Converting to SDF world")
            world.export_sdf(outputfile=args.output, mesh_format=args.mesh_type)
    elif args.output.lower().endswith("urdf"):
        log.info("Converting to URDF")
        robot.export_urdf(outputfile=args.output, mesh_format=args.mesh_type)
    elif args.output.lower().endswith("pdf"):
        log.info("Converting to PDF")
        if robot:
            robot.export_pdf(outputfile=args.output)
        else:
            raise NotImplementedError("Can't yet create PDF for world representations")
    elif args.output.lower().endswith("smurf") or path.isdir(args.output):
        log.info("Converting to SMURF")
        if robot:
            robot.export(outputdir=args.output, export_config=resources.get_default_export_config(), mesh_format=args.mesh_type)
        else: # todo this is also valid for sdf and urdf
            if not path.isdir(args.output):
                raise IOError(f"Please specify a directory as output, when exporting a world with multiple entities. {args.output} is not a directory.")
            for root in world.get_root_entities():
                robot = world.assemble(root)
                robot.export(outputdir=path.join(args.output, f"root-{root.name}"), export_config=getattr(world, "export_config", resources.get_default_export_config()), mesh_format=args.mesh_type)
    elif args.output.lower().endswith("jpg") or args.output.lower().endswith("png"):
        if robot:
            log.info("Creating thumbnail")
            size=512
            im = misc.get_thumbnail(robot.xmlfile, icon_size=size)
            misc.make_icon(im, args.output, size=size)
        else:
            raise NotImplementedError("Can't yet export thumbnails for world representations")
    elif args.output.lower().endswith("smurfs"):
        if robot:
            world = World()
            world.add_robot(name="robot", robot=robot, anchor="world")
        world.export_smurfs(outputfile=args.output, mesh_format=args.mesh_type)
    elif args.output.lower().endswith("smurfa"):
        if robot:
            world = World()
            world.add_robot(name="robot", robot=robot, anchor="world")
        world.export_smurfa(outputfile=args.output, mesh_format=args.mesh_type)
    else:
        log.error(f"Don't know which conversion to apply for {args.output}")
        return 1
    return 0


if __name__ == '__main__':
    import sys

    main(sys.argv)
