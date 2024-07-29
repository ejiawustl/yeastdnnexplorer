import logging

from shiny import module, reactive, ui

logger = logging.getLogger("shiny")


def make_api_call(username, password):
    # Replace this with actual API call logic
    if username == "user" and password == "pass":
        return {"status": "success", "token": "dummy_token"}
    else:
        return {"status": "fail", "message": "Invalid credentials"}


@module.ui
def login_interface_ui():
    return ui.div(
        ui.input_text("username", "Username"),
        ui.input_password("password", "Password"),
        ui.input_action_button("login_btn", "Login"),
        id="login_form",
    )


@module.server
def login_interface_server(input, output, session):
    _authenticated = reactive.Value(False)
    _token = reactive.Value(None)  # Reactive value to store the token

    @reactive.Effect
    @reactive.event(input.login_btn)
    def login_btn():
        username = input.username()
        password = input.password()

        logger.debug(f"login_btn: {username}, {password}")

        response = make_api_call(username, password)

        if response["status"] == "success":
            _authenticated.set(True)
            _token.set(response["token"])
            logger.debug(f"token: {_token.get()}")
        else:
            m = ui.modal(
                ui.p(response["message"]),
                title="Login Failed",
                easy_close=True,
                footer=ui.modal_button("Close"),
            )
            ui.modal_show(m)

    # Provide a way to access the token from outside the module
    return _authenticated, _token
