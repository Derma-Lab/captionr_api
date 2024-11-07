import argparse
import pathlib
import logging
from dataclasses import dataclass
from PIL import Image
import os
from captionr.clip_interrogator import Interrogator, Config
from captionr.captionr_class import CaptionrConfig, Captionr
from tqdm import tqdm

config: CaptionrConfig = None

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='Captionr',
        usage="%(prog)s [OPTIONS] [FOLDER]...",
        description="Caption a set of images"
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 0.0.1"
    )
    parser.add_argument('folder',
                        help='One or more folders to scan for images. Images should be jpg/png.',
                        type=pathlib.Path,
                        nargs='*',
                        )
    parser.add_argument('--output',
                        help='Output to a folder rather than side by side with image files',
                        type=pathlib.Path,
                        nargs=1
                        )
    parser.add_argument('--existing',
                        help='Action to take for existing caption files (default: skip)',
                        choices=['skip', 'ignore', 'copy', 'prepend', 'append'],
                        default='skip'
                        )
    parser.add_argument('--cap_length',
                        help='Maximum length of caption. (default: 0)',
                        default=0,
                        type=int
                        )
    parser.add_argument('--clip_model_name',
                        help='CLIP model to use. (default: ViT-H-14/laion2b_s32b_b79k)',
                        default='ViT-H-14/laion2b_s32b_b79k',
                        choices=['ViT-H-14/laion2b_s32b_b79k', 'ViT-L-14/openai', 'ViT-bigG-14/laion2b_s39b_b160k']
                        )
    parser.add_argument('--clip_flavor',
                        help='Add CLIP Flavors',
                        action='store_true'
                        )
    parser.add_argument('--clip_max_flavors',
                        help='Max CLIP Flavors (default: 8)',
                        default=8,
                        type=int
                        )
    parser.add_argument('--clip_artist',
                        help='Add CLIP Artists',
                        action='store_true'
                        )
    parser.add_argument('--clip_medium',
                        help='Add CLIP Mediums',
                        action='store_true'
                        )
    parser.add_argument('--clip_movement',
                        help='Add CLIP Movements',
                        action='store_true'
                        )
    parser.add_argument('--clip_trending',
                        help='Add CLIP Trendings',
                        action='store_true'
                        )
    parser.add_argument('--clip_method',
                        help='CLIP method to use',
                        choices=['interrogate', 'interrogate_fast', 'interrogate_classic'],
                        default='interrogate_fast'
                        )
    parser.add_argument('--ignore_tags',
                        help='Comma separated list of tags to ignore',
                        )
    parser.add_argument('--find',
                        help='Perform find and replace with --replace REPLACE',
                        )
    parser.add_argument('--replace',
                        help='Perform find and replace with --find FIND',
                        )
    parser.add_argument('--folder_tag',
                        help='Tag the image with folder name',
                        action='store_true'
                        )
    parser.add_argument('--folder_tag_levels',
                        help='Number of folder levels to tag. (default: 1)',
                        type=int,
                        default=1,
                        )
    parser.add_argument('--folder_tag_stop',
                        help='Do not tag folders any deeper than this path.',
                        type=pathlib.Path,
                        )
    parser.add_argument('--uniquify_tags',
                        help='Ensure tags are unique',
                        action='store_true'
                        )
    parser.add_argument('--fuzz_ratio',
                        help='Sets the similarity ratio for tag uniqueness. (default: 60.0)',
                        type=float,
                        default=60.0
                        )
    parser.add_argument('--prepend_text',
                        help='Prepend text to final caption',
                        )
    parser.add_argument('--append_text',
                        help='Append text to final caption',
                        )
    parser.add_argument('--preview',
                        help='Do not write to caption file. Just displays preview in STDOUT',
                        action='store_true'
                        )
    parser.add_argument('--use_filename',
                        help='Use filename as the initial caption, stripping special characters/numbers',
                        action='store_true'
                        )
    parser.add_argument('--device',
                        help='Device to use. (default: cuda)',
                        choices=['cuda', 'cpu'],
                        default='cuda'
                        )
    parser.add_argument('--extension',
                        help='Caption file extension. (default: txt)',
                        choices=['txt', 'caption'],
                        default='txt'
                        )
    parser.add_argument('--quiet',
                        action='store_true'
                        )
    parser.add_argument('--debug',
                        action='store_true'
                        )
    return parser

def main() -> None:
    global config
    parser = init_argparse()
    config = parser.parse_args()
    config.base_path = os.path.dirname(os.path.abspath(__file__))
    if config.debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug(config)
    elif config.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    if len(config.folder) == 0:
        parser.error('Folder is required.')

    if not config.clip_flavor and not config.clip_artist and not config.clip_medium \
            and not config.clip_movement and not config.clip_trending:
        if config.existing == 'skip' and (config.find or config.folder_tag or config.prepend_text or config.append_text):
            parser.error('--existing=skip cannot be used without specifying a caption model or modifications.')
        elif config.existing == 'skip':
            parser.error('No captioning flags specified. Use --clip_flavor | --clip_artist | --clip_medium | --clip_movement | --clip_trending | --find/--replace | --folder_tag | --prepend_text | --append_text to initiate captioning.')

    if config.clip_artist or config.clip_flavor or config.clip_medium or config.clip_movement or config.clip_trending:
        logging.info("Loading CLIP Model...")
        config._clip = Interrogator(Config(
            clip_model_name=config.clip_model_name,
            captionr_config=config,
            quiet=config.quiet,
            data_path=os.path.join(config.base_path, 'data'),
            cache_path=os.path.join(config.base_path, 'data')
        ))

    if config.preview:
        logging.info('PREVIEW MODE ENABLED. No caption files will be written.')

    paths = []
    cptr = Captionr(config=config)
    for folder in config.folder:
        for root, dirs, files in os.walk(folder.absolute(), topdown=False):
            for name in files:
                if os.path.splitext(os.path.basename(name))[1].upper() not in ['.JPEG', '.JPG', '.JPE', '.PNG']:
                    continue
                cap_file = os.path.join(os.path.dirname(os.path.join(root, name)), os.path.splitext(os.path.basename(name))[0] + f'.{config.extension}')
                if not config.existing == 'skip' or not os.path.exists(cap_file):
                    paths.append(os.path.join(root, name))
                elif not config.quiet:
                    logging.info(f'Caption file {cap_file} exists. Skipping.')
    for path in tqdm(paths):
        cptr.process_img(path)

if __name__ == "__main__":
    main()
