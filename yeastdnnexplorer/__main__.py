import argparse
import logging
from typing import Any

from shiny import run_app

from yeastdnnexplorer.utils import LogLevel, configure_logger


def configure_logging(log_level: int) -> tuple[logging.Logger, logging.Logger]:
    """
    Configure the logging for the application.

    :param log_level: The logging level to set.
    :return: A tuple of the main and shiny loggers.

    """
    main_logger = configure_logger("main", level=log_level)
    shiny_logger = configure_logger("shiny", level=log_level)
    return main_logger, shiny_logger


def run_shiny(args: argparse.Namespace) -> None:
    """
    Run the shiny app with the specified arguments.

    :param args: The parsed command-line arguments.

    """
    kwargs = {}
    if args.debug:
        kwargs["reload"] = True
        kwargs["reload_dirs"] = ["yeastdnnexplorer/shiny_app"]  # type: ignore
    app_import_string = "yeastdnnexplorer.shiny_app.app:app"
    run_app(app_import_string, **kwargs)


def run_another_command(args: argparse.Namespace) -> None:
    """
    Run another command with the specified arguments.

    :param args: The parsed command-line arguments.

    """
    print(f"Running another command with parameter: {args.param}")


class CustomHelpFormatter(argparse.HelpFormatter):
    """Custom help formatter to format the subcommands and general options sections."""

    def __init__(self, prog: str):
        super().__init__(prog)
        self.general_actions: list[argparse.Action] = []

    def add_general_arguments(self, actions: list[argparse.Action]) -> None:
        """
        Add general arguments to be displayed in a separate section.

        :param actions: The general actions to add.

        """
        self.general_actions.extend(actions)

    def add_arguments(self, actions: Any) -> None:
        """
        Add arguments to the help message, customizing for subcommands.

        :param actions: The actions to add.

        """
        # Filter out subparsers action to handle it separately
        actions_to_add = []
        subparsers_action = None
        for action in actions:
            if isinstance(action, argparse._SubParsersAction):
                subparsers_action = action
            else:
                actions_to_add.append(action)

        # Add non-subparser actions (options)
        super().add_arguments(actions_to_add)

        # Add subparsers action (subcommands)
        if subparsers_action:
            self._add_item(self._format_subparsers, [subparsers_action])

    def _format_subparsers(self, action: argparse._SubParsersAction) -> str:
        """
        Format the subcommands for the help message.

        :param action: The subparsers action.
        :return: A formatted string of subcommands.

        """
        parts = ["subcommands:"]
        for choice, subaction in action.choices.items():
            parts.append(f"  {choice:<20} {subaction.description}")
        return "\n".join(parts)

    def format_help(self) -> str:
        """
        Format the help message.

        :return: The formatted help message.

        """
        help_text = super().format_help()
        # Replace "positional arguments" section if present
        help_text = help_text.replace("positional arguments:", "subcommands:")
        # Ensure "subcommands:" is not doubled
        help_text = help_text.replace("subcommands:\nsubcommands:", "subcommands:")

        if self.general_actions:
            general_help_text = "\n\nGeneral options:\n"
            for action in self.general_actions:
                general_help_text += (
                    f"  {self._format_action_invocation(action)}\n    {action.help}\n"
                )
            help_text += general_help_text

        return help_text


def add_general_arguments_to_subparsers(subparsers, general_arguments):
    for subparser in subparsers.choices.values():
        for arg in general_arguments:
            subparser._add_action(arg)


def main() -> None:
    """Main entry point for the YeastDNNExplorer application."""
    parser = argparse.ArgumentParser(
        prog="yeastdnnexplorer",
        description="YeastDNNExplorer Main Entry Point",
        usage="yeastdnnexplorer --help",
        formatter_class=CustomHelpFormatter,
    )

    # Create an instance of CustomHelpFormatter to access
    # the add_general_arguments method
    formatter = parser._get_formatter()

    # Shared parameter for logging level
    log_level_argument = parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    formatter.add_arguments([log_level_argument])

    # Define subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Shiny command with a debug flag
    shiny_parser = subparsers.add_parser(
        "shiny",
        help="Run the shiny app",
        description="Run the shiny app",
        usage="yeastdnnexplorer shiny [-h] [--debug]",
        formatter_class=CustomHelpFormatter,
    )
    shiny_parser.add_argument(
        "--debug", action="store_true", help="Run the app with reloading enabled"
    )
    shiny_parser.set_defaults(func=run_shiny)

    # Another command with its own parameter
    another_parser = subparsers.add_parser(
        "another_command",
        help="Run another command",
        description="Run another command",
        usage="yeastdnnexplorer another_command --param PARAM",
        formatter_class=CustomHelpFormatter,
    )
    another_parser.add_argument(
        "--param", type=str, required=True, help="A parameter for another command"
    )
    another_parser.set_defaults(func=run_another_command)

    # Add the general arguments to the subcommand parsers
    add_general_arguments_to_subparsers(subparsers, [log_level_argument])

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    try:
        log_level = LogLevel.from_string(args.log_level)
    except ValueError as e:
        print(e)
        parser.print_help()
        return

    main_logger, shiny_logger = configure_logging(log_level)

    # Run the appropriate command
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
