import PySimpleGUI as sg


def boolean_question(question: str):
    layout_column = [
        [sg.Text(question, size=(40, 1), font=("Any 15"), justification="center")],
        [sg.Button("Yes", size=(40, 1)), sg.Button("No", size=(40, 1))],
    ]
    layout = [[sg.Column(layout_column, element_justification="center")]]

    window = sg.Window("Reconfiguration", layout, location=(800, 400), finalize=True)
    move_center(window)

    while True:
        event, values = window.read(timeout=20)

        if event == "Exit" or event == sg.WIN_CLOSED:
            exit(0)

        if event == "Yes":
            window.Close()
            return True

        if event == "No":
            window.Close()
            return False

    window.Close()
    return True


def move_center(window):
    screen_width, screen_height = window.get_screen_dimensions()
    win_width, win_height = window.size
    x, y = (screen_width - win_width) // 2, (screen_height - win_height) // 2
    window.move(x, y)
