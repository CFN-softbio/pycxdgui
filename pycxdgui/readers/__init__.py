# register the readers and make the singleton
from .readers_extensions import extension_dict
from .readers_base import ReaderRegistry

main_reader = ReaderRegistry(extension_dict=extension_dict)
