import sys, os
from airfoil_converter import dat_to_CST
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTextEdit, QPushButton, QLineEdit, QGroupBox,
    QFormLayout, QStackedWidget, QCheckBox, QToolButton, QFrame, 
    QProgressBar, QShortcut
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWebEngineWidgets import QWebEngineView

# add the parent folder of adero/ to the module search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adero.input_panel import UserInputPanel
from adero.main_processes import MainProcesses

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.wing_params = None
        self.processes = MainProcesses(self)
        self.setWindowTitle("Wing Geometry Generator")
        self._build_ui()
        self._install_shortcuts()

    def _build_ui(self):
        
        # Input panel
        self.input_panel = UserInputPanel()
        # Log output
        self.script_reader = QTextEdit(readOnly=True)
        self.script_reader.append("System Log:\n")

        # 3D and 2D viewers
        self.viewer_3d = QWebEngineView()
        self.viewer_3d.setHtml("<h3>3D Geometry Viewer</h3>")
        self.viewer_2d = QWebEngineView()
        self.viewer_2d.setHtml("<h3>2D Airfoil Plot</h3>")
        self.viewer_stack = QStackedWidget()
        self.viewer_stack.addWidget(self.viewer_3d)
        self.viewer_stack.addWidget(self.viewer_2d)

        # View options toggle
        self._build_view_options()

        # Control buttons: make a named run_button
        btn_layout = QHBoxLayout()

        # replace the old loop-entry for "Run" with:
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.processes.run)
        btn_layout.addWidget(self.run_button)

        # keep Reset and Save as before
        for name, handler in [("Reset", self.processes.reset),
                              ("Save",  self.processes.save)]:
            btn = QPushButton(name)
            btn.clicked.connect(handler)
            btn_layout.addWidget(btn)

        # … then later, after you've created self.input_panel …
        # whenever the user flips between Flight / Component modes:
        self.input_panel.switcher.stack.currentChanged.connect(self._on_mode_changed)

        # create a progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.toggle_button)
        right_layout.addWidget(self.toggle_frame)
        right_layout.addWidget(self.viewer_stack)
        right_layout.addLayout(btn_layout)
        right_layout.addWidget(self.progress)
        right_layout.addWidget(self.script_reader)

        splitter = QSplitter(Qt.Horizontal)
        left_container = QWidget()
        left_container.setLayout(QVBoxLayout())
        left_container.layout().addWidget(self.input_panel)

        right_container = QWidget()
        right_container.setLayout(right_layout)

        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setSizes([250, 750])

        central = QWidget()
        central.setLayout(QVBoxLayout())
        central.layout().addWidget(splitter)
        self.setCentralWidget(central)
    
    def _on_mode_changed(self, idx):
        if idx == 0:
            self.run_button.setText("Run")
        else:
            self.run_button.setText("Predict")

    def _install_shortcuts(self):
        # Make Return (and numeric keypad Enter) fire the same .run() slot
        enter_shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        enter_shortcut.activated.connect(self.processes.run)
        kp_enter_shortcut = QShortcut(QKeySequence(Qt.Key_Enter), self)
        kp_enter_shortcut.activated.connect(self.processes.run)

    def _build_view_options(self):
        self.toggle_button = QToolButton(text="View Options", checkable=True)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.clicked.connect(self._toggle_options)

        self.toggle_frame = QFrame(visible=False)
        layout = QVBoxLayout(self.toggle_frame)
        self.btn_3d = QPushButton("Show 3D")
        self.btn_2d = QPushButton("Show 2D")
        self.chk_upper = QCheckBox("Show Upper Surface")
        self.chk_lower = QCheckBox("Show Lower Surface")
        self.chk_internal = QCheckBox("Show Internal Structure")
        self.chk_upper.setChecked(True)
        self.chk_lower.setChecked(True)
        self.chk_internal.setChecked(True)

        self.btn_3d.clicked.connect(lambda: self.viewer_stack.setCurrentIndex(0))
        self.btn_2d.clicked.connect(lambda: self.viewer_stack.setCurrentIndex(1))

        for widget in (self.btn_3d, self.btn_2d,
                       self.chk_upper, self.chk_lower, self.chk_internal):
            layout.addWidget(widget)

        self.chk_upper.stateChanged.connect(self._refresh_view)
        self.chk_lower.stateChanged.connect(self._refresh_view)
        self.chk_internal.stateChanged.connect(self._refresh_view)


    def _refresh_view(self):
        import tempfile
        import plotly.io as pio
        from adero.cst import cst_model
        from adero.visualizer import generate_wing_figure, generate_2d_airfoil_plot

        show_i = self.chk_internal.isChecked()
        upper_op = 1 if self.chk_upper.isChecked() else 0
        lower_op = 1 if self.chk_lower.isChecked() else 0
        fig3d = generate_wing_figure(
            self.wing_params,
            upper_opacity = upper_op,
            lower_opacity = lower_op,
            show_internal = self.chk_internal.isChecked()
        )
        html3 = pio.to_html(fig3d, full_html=True).encode('utf-8')
        tmp3  = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        tmp3.write(html3); tmp3.close()
        self.viewer_3d.load(QUrl.fromLocalFile(tmp3.name))

        fig2d = generate_2d_airfoil_plot(self.wing_params)
        html2 = pio.to_html(fig2d, full_html=True).encode('utf-8')
        tmp2  = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        tmp2.write(html2); tmp2.close()
        self.viewer_2d.load(QUrl.fromLocalFile(tmp2.name))


    def _toggle_options(self):
        expanded = self.toggle_button.isChecked()
        self.toggle_frame.setVisible(expanded)
        self.toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1100, 700)
    window.show()
    sys.exit(app.exec_())