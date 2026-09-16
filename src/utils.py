def format_metadata(description, parameters, return_type, library, keywords, param_count):
    """
    Combines metadata fields into a single space-separated string for the model pipeline.
    """
    return f"{description} {parameters} {return_type} {library} {keywords} {param_count}"
