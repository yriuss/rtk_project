import warnings


def deprecated(alternative):
    def decorator(func):
        def new_func(*args, **kwargs):
            # Raise a DeprecationWarning with the specified message.
            message = f"Call to deprecated function {func.__name__}."
            if alternative:
                message += f" Use {alternative} instead."
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return new_func
    return decorator
