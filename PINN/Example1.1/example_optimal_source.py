from argparse import ArgumentParser
import yaml

from Nonsys.Nonsys.problem_nonlineardiffusion import NonsysElliptical
from Nonsys.Nonsys.model import DeepNonsysElliptical

def main():
    # parameters
    parser = ArgumentParser(description="Basic paser")
    parser.add_argument("--config_path", type=str,
                        help="Path to the configuration file")
    args = parser.parse_args()
    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    print(config)

    # model
    ndim = config["solution_net_params"]["net_dim"]
    problem = NonsysElliptical(ndim, config=config)
    model = DeepNonsysElliptical(problem=problem, config=config)
    model.train()


if __name__ == '__main__':
    main()