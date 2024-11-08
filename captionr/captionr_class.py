import pathlib
import logging
from dataclasses import dataclass
from PIL import Image
import os
from captionr.clip_interrogator import Interrogator, Config
import torch
import re
from thefuzz import fuzz

@dataclass
class CaptionrConfig:
    folder = None
    output: pathlib.Path = None
    existing = 'skip'
    cap_length = 150
    clip_model_name = 'ViT-H-14/laion2b_s32b_b79k'
    clip_flavor = False
    clip_max_flavors = 8
    clip_artist = False
    clip_medium = False
    clip_movement = False
    clip_trending = False
    clip_method = 'interrogate_fast'
    ignore_tags = ''
    find = ''
    replace = ''
    folder_tag = False
    folder_tag_levels = 1
    folder_tag_stop: pathlib.Path = None
    preview = False
    use_filename = False
    append_text = ''
    prepend_text = ''
    uniquify_tags = False
    device = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    extension = 'txt'
    quiet = False
    debug = False
    base_path = os.path.dirname(__file__)
    fuzz_ratio = 60.0
    _clip: Interrogator = None

class Captionr:
    def __init__(self, config: CaptionrConfig) -> None:
        self.config = config

    def get_parent_folder(self, filepath, levels=1):
        common = os.path.split(filepath)[0]
        paths = []
        for i in range(int(levels)):
            split = os.path.split(common)
            common = split[0]
            paths.append(split[1])
            if self.config.folder_tag_stop is not None and \
                    self.config.folder_tag_stop != '' and \
                    split[0] == self.config.folder_tag_stop:
                break
        return paths

    def process_img_api(self, img):
        config = self.config
        try:
            # Since we're processing an image directly, no file operations are needed
            existing_caption = ''
            new_caption = existing_caption

            # Use clip_interrogator to process image and existing caption
            if (config.clip_artist or config.clip_flavor or config.clip_trending or config.clip_movement or config.clip_medium) and config._clip is not None:
                func = getattr(config._clip, config.clip_method)
                tags = func(caption=new_caption, image=img, max_flavors=config.clip_max_flavors)
                logging.debug(f'CLIP tags: {tags}')
                out_tags = [tag.strip() for tag in tags.split(",")]
            else:
                out_tags = []

            # Remove duplicates, filter similar tags
            unique_tags = []
            tags_to_ignore = []
            if config.ignore_tags != "" and config.ignore_tags is not None:
                si_tags = config.ignore_tags.split(",")
                for tag in si_tags:
                    tags_to_ignore.append(tag.strip())

            if config.uniquify_tags:
                for tag in out_tags:
                    tstr = tag.strip()
                    if not tstr in unique_tags and not "_\(" in tag and tstr not in tags_to_ignore:
                        should_append = True
                        for s in unique_tags:
                            if fuzz.ratio(s, tstr) > self.config.fuzz_ratio:
                                should_append = False
                                break
                        if should_append:
                            unique_tags.append(tag.replace('"', '').strip())
            else:
                for tag in out_tags:
                    if not "_\(" in tag and tag.strip() not in tags_to_ignore:
                        unique_tags.append(tag.replace('"', '').strip())

            # Construct new caption from tag list
            caption_txt = ", ".join(unique_tags)

            if config.find is not None and config.find != '' and config.replace is not None and config.replace != '':
                if f"{config.find}" in caption_txt:
                    caption_txt = caption_txt.replace(f"{config.find}", config.replace)

            tags = caption_txt.split(" ")
            if config.cap_length != 0 and len(tags) > config.cap_length:
                tags = tags[0:config.cap_length]
                tags[-1] = tags[-1].rstrip(",")
            caption_txt = " ".join(tags)

            if config.append_text != '' and config.append_text is not None:
                caption_txt = caption_txt + config.append_text

            if config.prepend_text != '' and config.prepend_text is not None:
                caption_txt = config.prepend_text.rstrip().lstrip() + ' ' + caption_txt

            return caption_txt
        except Exception as e:
            logging.exception(f"Exception occurred processing image")
            raise e

    def process_img(self, img_path):
        config = self.config
        try:
            # Load image
            with Image.open(img_path).convert('RGB') as img:
                # Get existing caption
                existing_caption = ''
                cap_file = os.path.join(os.path.dirname(img_path), os.path.splitext(os.path.basename(img_path))[0] + f'.{config.extension}')
                if os.path.isfile(cap_file):
                    try:
                        with open(cap_file) as f:
                            existing_caption = f.read()
                    except Exception as e:
                        logging.exception(f"Got exception reading caption file: {e}")

                # Get caption from filename if empty
                if existing_caption == '' and config.use_filename:
                    path = os.path.basename(img_path)
                    path = os.path.splitext(path)[0]
                    existing_caption = ''.join(c for c in path if c.isalpha() or c in [" ", ","])

                # Initialize out_tags with existing caption
                out_tags = []
                new_caption = existing_caption

                # Use clip_interrogator to process image and existing caption
                if (config.clip_artist or config.clip_flavor or config.clip_trending or config.clip_movement or config.clip_medium) and config._clip is not None:
                    func = getattr(config._clip, config.clip_method)
                    tags = func(caption=new_caption, image=img, max_flavors=config.clip_max_flavors)
                    logging.debug(f'CLIP tags: {tags}')
                    for tag in tags.split(","):
                        out_tags.append(tag.strip())
                else:
                    for tag in new_caption.split(","):
                        out_tags.append(tag.strip())

                # Add parent folder to tag list if enabled
                if config.folder_tag:
                    folder_tags = self.get_parent_folder(img_path, config.folder_tag_levels)
                    for tag in folder_tags:
                        out_tags.append(tag.strip())

                # Remove duplicates, filter similar tags
                unique_tags = []
                tags_to_ignore = []
                if config.ignore_tags != "" and config.ignore_tags is not None:
                    si_tags = config.ignore_tags.split(",")
                    for tag in si_tags:
                        tags_to_ignore.append(tag.strip())

                if config.uniquify_tags:
                    for tag in out_tags:
                        tstr = tag.strip()
                        if not tstr in unique_tags and not "_\(" in tag and tstr not in tags_to_ignore:
                            should_append = True
                            for s in unique_tags:
                                if fuzz.ratio(s, tstr) > self.config.fuzz_ratio:
                                    should_append = False
                                    break
                            if should_append:
                                unique_tags.append(tag.replace('"', '').strip())
                else:
                    for tag in out_tags:
                        if not "_\(" in tag and tag.strip() not in tags_to_ignore:
                            unique_tags.append(tag.replace('"', '').strip())

                existing_tags = existing_caption.split(",")
                logging.debug(f'Unique tags: {unique_tags}')
                logging.debug(f'Existing Tags: {existing_tags}')

                # Handle existing captions based on the specified option
                if config.existing == "prepend" and len(existing_tags):
                    new_tags = existing_tags
                    for tag in unique_tags:
                        if not tag.strip() in new_tags or not config.uniquify_tags:
                            new_tags.append(tag.strip())
                    unique_tags = new_tags

                if config.existing == 'append' and len(existing_tags):
                    for tag in existing_tags:
                        if not tag.strip() in unique_tags or not config.uniquify_tags:
                            unique_tags.append(tag.strip())

                if config.existing == 'copy' and existing_caption:
                    for tag in existing_tags:
                        unique_tags.append(tag.strip())

                try:
                    unique_tags.remove('')
                except ValueError:
                    pass

                # Construct new caption from tag list
                caption_txt = ", ".join(unique_tags)

                if config.find is not None and config.find != '' and config.replace is not None and config.replace != '':
                    if f"{config.find}" in caption_txt:
                        caption_txt = caption_txt.replace(f"{config.find}", config.replace)

                tags = caption_txt.split(" ")
                if config.cap_length != 0 and len(tags) > config.cap_length:
                    tags = tags[0:config.cap_length]
                    tags[-1] = tags[-1].rstrip(",")
                caption_txt = " ".join(tags)

                if config.append_text != '' and config.append_text is not None:
                    caption_txt = caption_txt + config.append_text

                if config.prepend_text != '' and config.prepend_text is not None:
                    caption_txt = config.prepend_text.rstrip().lstrip() + ' ' + caption_txt

                # Write caption file
                if not config.preview:
                    if config.output == '' or config.output is None:
                        dirname = os.path.dirname(cap_file)
                    else:
                        dirname = str(config.output[0]) if isinstance(config.output, list) else str(config.output)
                    outputfilename = os.path.join(dirname, os.path.basename(cap_file))
                    with open(outputfilename, "w", encoding="utf8") as file:
                        file.write(caption_txt)
                        logging.debug(f'Wrote {outputfilename}')

                if config.preview:
                    logging.info(f'PREVIEW: {caption_txt}')
                    logging.info('No caption file written.')
                else:
                    logging.info(f'{outputfilename}: {caption_txt}')

                return caption_txt
        except Exception as e:
            logging.exception(f"Exception occurred processing {img_path}")
