import shutil
import textwrap


# region colorhelpers
def terminal_font_color(red: int, green: int, blue: int) -> str:
    """
    Return the ANSI escape code for a terminal font color.

    Parameters
    ----------
    red : int
        The red value.
    green : int
        The green value.
    blue : int
        The blue value.

    Returns
    -------
    str
        _description_
    """
    return f"\u001b[38;2;{red};{green};{blue}m"


def terminal_bg_color(red: int, green: int, blue: int) -> str:
    """
    Return the ANSI escape code for a terminal background color.

    Parameters
    ----------
    red : int
        The red value.
    green : int
        The green value.
    blue : int
        The blue value.

    Returns
    -------
    str
        The ANSI escape code for the terminal background color.
    """
    return f"\u001b[48;2;{red};{green};{blue}m"


# endregion

RESET = "\u001b[0m"
BOLD = "\u001b[1m"
FAINT = "\u001b[2m"
ITALIC = "\u001b[3m"
UNDERLINED = "\u001b[4m"
INVERSE = "\u001b[7m"
STRIKETHROUGH = "\u001b[9m"
# https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
# https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_(Select_Graphic_Rendition)_parameters
# for other ANSI codes

TERMINAL_WIDTH = shutil.get_terminal_size().columns - 24
VERBATIM = "\u0000"
NEWLINE = "\u000D\u000A\u0003"
# region Color Formatting
BLACK = "\u001b[30m"
RED = "\u001b[31m"
GREEN = "\u001b[32m"
YELLOW = "\u001b[33m"
BLUE = "\u001b[34m"
MAGENTA = "\u001b[35m"
CYAN = "\u001b[36m"
WHITE = "\u001b[37m"
BRIGHT_BLACK = "\u001b[90m"
BRIGHT_RED = "\u001b[91m"
BRIGHT_GREEN = "\u001b[92m"
BRIGHT_YELLOW = "\u001b[93m"
BRIGHT_BLUE = "\u001b[94m"
BRIGHT_MAGENTA = "\u001b[95m"
BRIGHT_CYAN = "\u001b[96m"
BRIGHT_WHITE = "\u001b[97m"
BLACK_BG = "\u001b[40m"
RED_BG = "\u001b[41m"
GREEN_BG = "\u001b[42m"
YELLOW_BG = "\u001b[43m"
BLUE_BG = "\u001b[44m"
MAGENTA_BG = "\u001b[45m"
CYAN_BG = "\u001b[46m"
WHITE_BG = "\u001b[47m"
BRIGHT_BLACK_BG = "\u001b[100m"
BRIGHT_RED_BG = "\u001b[101m"
BRIGHT_GREEN_BG = "\u001b[102m"
BRIGHT_YELLOW_BG = "\u001b[103m"
BRIGHT_BLUE_BG = "\u001b[104m"
BRIGHT_MAGENTA_BG = "\u001b[105m"
BRIGHT_CYAN_BG = "\u001b[106m"
BRIGHT_WHITE_BG = "\u001b[107m"
# endregion

VERBATIM_FORMAT = GREEN


def stringformat(text: str) -> str:
    """
    Format a string to be displayed in the terminal for help messages. Includes
    word wrapping, color formatting, and enforced line breaks. This is a
    helper method essentially just for my own usage (but feel free to use it
    in your own stuff if you want).

    Parameters
    ----------
    text : str
        The text to format.

    Returns
    -------
    str
        The formatted text.
    """
    newline_split = text.split(textwrap.dedent(NEWLINE))
    for i in range(len(newline_split)):
        stripped = newline_split[i].strip()
        stripped = stripped.replace(VERBATIM, VERBATIM_FORMAT)
        # Reset color scheme after newline
        stripped += RESET
        newline_split[i] = textwrap.fill(stripped, width=TERMINAL_WIDTH)
    return "\n".join(newline_split)
