"""Parse image names and encode image-pair keys."""

from pathlib import Path


def names_to_pair(name0, name1, separator="/"):
    return separator.join((name0.replace("/", "-"), name1.replace("/", "-")))


def parse_image_list(path):
    images = []
    with open(path) as f:
        for line in f:
            line = line.strip("\n")
            if len(line) == 0 or line[0] == "#":
                continue
            name, *_ = line.split()
            images.append(name)

    assert len(images) > 0
    print(f"Imported {len(images)} images from {path.name}")
    return images


def parse_image_lists(paths):
    images = []
    files = list(Path(paths.parent).glob(paths.name))
    assert len(files) > 0
    for lfile in files:
        images += parse_image_list(lfile)
    return images
