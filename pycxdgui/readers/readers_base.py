# idea taken from filestore (NSLS-II)
# just so it can work with "with" statement in python


class ReaderBase:
    extension = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # need to implement this
    def close(self):
        pass


class ReaderRegistry:
    ''' This handles all the reading writing of files.
        It does a search based on extension.
    '''
    def __init__(self, extension_dict=None):
        if extension_dict is None:
            extension_dict= dict()
        self.extension_dict = extension_dict

    @property
    def reader_dict(self):
        ''' the reader dict is the reverse dict of the extension
        dict. Useful for making file extension filters.'''
        return group_items(self.extension_dict)

    def get_file_filters(self, extension=None):
        return make_file_filters(self.extension_dict, extension=extension)

    def register(self, key, reader):
        if key in self.extension_dict:
            print("Warning, overwriting reader")

        self.extension_dict[key] = reader

    def _get_key_from_filename(self, filename):
        ''' Get a reader from a certain filename.'''
        _final_key = None
        for key in self.extension_dict.keys():
            if filename.endswith(key):
                _final_key = key
                break

        return _final_key

    def _get_reader_from_filename(self, filename):
        key = self._get_key_from_filename(filename)
        if key is None:
            raise ValueError("Error, key not found (register key using " +
                             "register_reader")
        reader = self.extension_dict[key]
        return reader

    def load_file(self, filename):
        ''' This is the main entrance to reading the file.
            This should be a singleton that figures out which
            object to use to open a file, and calls the respective reader.
            '''
        reader = self._get_reader_from_filename(filename)

        return reader(filename)

    def __call__(self, filename):
        return self.load_file(filename)


def make_file_filters(extension_dict, extension=None):
    ''' make a file filter for the file filter dialog
        from a reader registry reader_reg.
        If extension is set, this will try to make that extension come first
    '''
    filter_groups = group_items(extension_dict).copy()
    filter_string = ""
    first = True

    if extension is not None:
        # first locate first extension
        for reader, filt_list in filter_groups.items():
            if extension in filt_list:
                filter_string = filter_string + reader.description
                filter_string = filter_string + reader.description + " ("
                for filt in filter_groups[reader]:
                    filter_string = filter_string + filt + " "
                filter_string = filter_string + ")"
                first = False

    for reader, filt_list in filter_groups.items():
        if first is False:
            filter_string = filter_string + ";;"
        else:
            first = False
        filter_string = filter_string + reader.description + " ("
        for filt in filt_list:
            filter_string = filter_string + filt + " "
        filter_string = filter_string + ")"

    if first is False:
        filter_string = filter_string + ";;"

    # add All files option
    filter_string = filter_string + "All Files (*)"

    return filter_string


def group_items(dd):
    ''' group dict by value.
        useful for making file filters
    '''
    newdd = dict()
    # extension, reader
    for key, val in dd.items():
        if val in newdd:
            newdd[val].append(key)
        else:
            newdd[val] = [key]
    # becomes: reader : [extensions]
    return newdd
