import pya

# Enter your Python code here ..


class ScreenshotDialog(pya.QDialog):
    """
    This class implements a dialog with a screenshot display area and a
    screenshot button
    """

    def button_clicked(self, checked):
        """ Event handler: "Screenshot" button clicked """

        view = pya.Application.instance().main_window().current_view()

        # get the screenshot and place it in the image label
        if view is not None:
            self.image.setPixmap(pya.QPixmap.fromImage(view.get_image(4000, 4000)))
        else:
            self.image.setText("No layout opened to take screenshot from")

    def __init__(self, parent=None):
        """ Dialog constructor """

        super(ScreenshotDialog, self).__init__()

        self.setWindowTitle("Screenshot Saver")

        self.resize(400, 120)

        layout = pya.QVBoxLayout(self)
        self.setLayout(layout)

        self.image = pya.QLabel("Press the button to fetch a screenshot", self)
        layout.addWidget(self.image)

        button = pya.QPushButton('Screenshot', self)
        button.setFont(pya.QFont('Times', 18, pya.QFont.Bold))
        layout.addWidget(button)

        # attach the event handler
        button.clicked(self.button_clicked)

# Instantiate the dialog and make it visible initially.
# Passing the main_window will make it stay on top of the main window.
dialog = ScreenshotDialog(pya.Application.instance().main_window())
dialog.show()
