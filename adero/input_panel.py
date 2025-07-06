from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QGroupBox, QLineEdit, QPushButton, QStackedWidget, QTabWidget, QComboBox
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtCore import Qt
from airfoil_converter import dat_to_CST
import os
import tempfile
import plotly.io as pio
from PyQt5.QtWebEngineWidgets import QWebEngineView
from adero.visualizer import generate_2d_airfoil_plot


class FlightMissionInput(QWidget):
    def __init__(self):
        super().__init__()
        
        # --- build each group exactly as before ---
        # 1) mission profile
        form_m = QFormLayout()
        self.fields = {}
        for label, key in [
            ("Maximum Take-Off Weight (kg)", 'mtow'),
            ("Payload Mass (kg)", 'payload_mass'),
            ("Payload Volume (m³)", 'payload_volume'),
            ("Cruise Velocity (m/s)", 'cruise_velocity'),
            ("Cruise Altitude (m)", 'cruise_altitude'),
            ("Ceiling Altitude (m)", 'ceiling_altitude')
        ]:
            le = QLineEdit()
            fm = QFontMetrics(le.font())
            le.setMinimumWidth(fm.horizontalAdvance("0000000") + 12)
            self.fields[key] = le
            form_m.addRow(f"{label}:", le)
        group_m = QGroupBox("Mission Profile")
        group_m.setLayout(form_m)

        # 2) airfoil weights
        form_a = QFormLayout()
        form_a.setFormAlignment(Qt.AlignRight)
        form_a.setLabelAlignment(Qt.AlignRight)
        self.weights = {
            'lift': QLineEdit("0.2"),
            'drag': QLineEdit("0.2"),
            'lift_at_stall': QLineEdit("0.4"),
            'aerofoil_area': QLineEdit("0.2")
        }
        for k,w in self.weights.items():
            form_a.addRow(f"{k.replace('_',' ').capitalize()}:", w)
        group_a = QGroupBox("Airfoil Optimizer Weights")
        group_a.setLayout(form_a)

        # 3) wing-box weights
        form_wb = QFormLayout()
        form_wb.setFormAlignment(Qt.AlignRight)
        form_wb.setLabelAlignment(Qt.AlignRight)
        self.wb_weights = {
            'tip_deflection':     QLineEdit("0.238"),
            'max_stress':         QLineEdit("0.238"),
            'stress_ratio':       QLineEdit("0.036"),
            'root_bending_moment':QLineEdit("0.238"),
            'aspect_ratio':       QLineEdit("0.25")
        }
        for k,w in self.wb_weights.items():
            form_wb.addRow(f"{k.replace('_',' ').capitalize()}:", w)
        group_wb = QGroupBox("Wing-Box Optimizer Weights")
        group_wb.setLayout(form_wb)

        # 4) planform weights
        form_pf = QFormLayout()
        form_pf.setFormAlignment(Qt.AlignRight)
        form_pf.setLabelAlignment(Qt.AlignRight)
        self.planform_weights = {
            'lift':    QLineEdit("0.15"),
            'lift_minimum':  QLineEdit("0.15"),
            'drag_minimum':  QLineEdit("0.4"),
            'drag_at_lift_minimum': QLineEdit("0.3")
        }
        for k,w in self.planform_weights.items():
            form_pf.addRow(f"{k.replace('_',' ').capitalize()}:", w)
        group_pf = QGroupBox("Planform Optimizer Weights")
        group_pf.setLayout(form_pf)

        # 5) section weights
        form_s = QFormLayout()
        form_s.setFormAlignment(Qt.AlignRight)
        form_s.setLabelAlignment(Qt.AlignRight)
        self.sec_weights = {
            'total_deformation':  QLineEdit("0.2"),
            'shear_stress_(xy)':    QLineEdit("0.15"),
            'mass':               QLineEdit("0.15"),
            'rib_stress':         QLineEdit("0.1"),
            'spar_stress':        QLineEdit("0.1"),
            'airfoil_stress':     QLineEdit("0.1"),
            'root_bending_moment':QLineEdit("0.2")
        }
        for k,w in self.sec_weights.items():
            form_s.addRow(f"{k.replace('_',' ').capitalize()}:", w)
        group_s = QGroupBox("Section Optimizer Weights")
        group_s.setLayout(form_s)

        # --- now pack them into tabs ---
        tabs = QTabWidget()
        tabs.addTab(group_m,  "Mission")
        tabs.addTab(group_a,  "Airfoil Wts")
        tabs.addTab(group_wb, "Wing-Box Wts")
        tabs.addTab(group_pf, "Planform Wts")
        tabs.addTab(group_s,  "Section Wts")

        # final layout
        v = QVBoxLayout(self)
        v.addWidget(tabs)


    def get_inputs(self):
        """
        Exactly as before, returns five dicts in the same order.
        """
        mission = {k: w.text() for k, w in self.fields.items()}
        aero_w  = {k: w.text() for k, w in self.weights.items()}
        plan_w  = {k: w.text() for k, w in self.planform_weights.items()}
        wb_w    = {k: w.text() for k, w in self.wb_weights.items()}
        sec_w   = {k: w.text() for k, w in self.sec_weights.items()}
        return mission, aero_w, plan_w, wb_w, sec_w

class AirfoilInput(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.airfoil_selector = QComboBox()
        self.airfoil_selector.addItem("Custom Input")
        self._populate_airfoil_dropdown()
        self.airfoil_selector.currentIndexChanged.connect(self._load_selected_airfoil)
        form.addRow("Select Airfoil (.dat):", self.airfoil_selector)

        self.fields = {}
        for key in ['Re', 'Upper_CST_1','Upper_CST_2','Upper_CST_3',
                    'Lower_CST_1','Lower_CST_2','Lower_CST_3']:
            le = QLineEdit()
            self.fields[key] = le
            form.addRow(f"{key}:", le)

        layout.addLayout(form)

        # Web viewer for 2D plot
        self.plot_view = QWebEngineView()
        self.plot_view.setHtml("<h4>Airfoil CST Plot</h4>")
        layout.addWidget(self.plot_view)

    def get_inputs(self):
        return {k: w.text() for k,w in self.fields.items()}

    def _populate_airfoil_dropdown(self):
        airfoil_dir = os.path.join(os.getcwd(), 'adero/airfoils')
        if not os.path.exists(airfoil_dir):
            return
        for file in sorted(os.listdir(airfoil_dir)):
            if file.endswith(".dat"):
                name = os.path.splitext(file)[0]
                self.airfoil_selector.addItem(name)

    def _load_selected_airfoil(self):
        selection = self.airfoil_selector.currentText()
        if selection == "Custom Input":
            return

        airfoil_dir = os.path.join(os.getcwd(), "adero", "airfoils")
        path = os.path.join(airfoil_dir, selection + ".dat")
        try:
            upper, lower = dat_to_CST(path)

            # — populate the six CST fields all at once —
            #   keys are 'Upper_CST_1','Upper_CST_2','Upper_CST_3','Lower_CST_1','Lower_CST_2','Lower_CST_3'
            for i, coeff in enumerate(upper, start=1):
                self.fields[f"Upper_CST_{i}"].setText(f"{coeff:.6f}")
            for i, coeff in enumerate(lower, start=1):
                self.fields[f"Lower_CST_{i}"].setText(f"{coeff:.6f}")

            # (optional) print to console so you can see it firing:
            print(f"Loaded {selection}: upper={upper}, lower={lower}")

            # — now regenerate the 2D CST plot and push it into the main window —
            wing_params = {"CST_coeff": {"upper": upper, "lower": lower}}
            fig = generate_2d_airfoil_plot(wing_params)
            html = pio.to_html(fig, full_html=True).encode("utf-8")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            tmp.write(html)
            tmp.close()

            # Inject into MainWindow’s 2D viewer:
            mw = self.window()
            mw.viewer_2d.load(QUrl.fromLocalFile(tmp.name))
            mw.viewer_stack.setCurrentIndex(1)

        except Exception as e:
            print(f"Error loading airfoil '{selection}': {e}")


class WingBoxInput(QWidget):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self.fields = {}
        for key in ['Chord','Span','Taper','tc_avg','Wing_Loading']:
            le = QLineEdit()
            self.fields[key] = le
            form.addRow(f"{key}:", le)
    def get_inputs(self):
        return {k: w.text() for k,w in self.fields.items()}

class PlanformInput(QWidget):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self.fields = {}
        # AR, Area, Taper, Sweep, then 6 CST coeffs
        for key in ['AR','Area','Taper','Sweep'] \
                   + [f"Up_CST_{i}" for i in (0,1,2)] \
                   + [f"Lo_CST_{i}" for i in (0,1,2)]:
            le = QLineEdit()
            self.fields[key] = le
            form.addRow(f"{key}:", le)
    def get_inputs(self):
        return {k: w.text() for k,w in self.fields.items()}

class StructuralInput(QWidget):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self.fields = {}
        # Rib_Th, Spar_Th, Skin_Th, No_Ribs, Root_Chord, Tip_Chord, Span, Sweep
        for key in ['Rib_Th','Spar_Th','Skin_Th','No_Ribs',
                    'Root_Chord','Tip_Chord','Span','Sweep']:
            le = QLineEdit()
            self.fields[key] = le
            form.addRow(f"{key}:", le)
    def get_inputs(self):
        return {k: w.text() for k,w in self.fields.items()}

class ComponentPredictionPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        self.panels = {
            'airfoil': AirfoilInput(),
            'wingbox': WingBoxInput(),
            'planform': PlanformInput(),
            'structural': StructuralInput()
        }
        tabs.addTab(self.panels['airfoil'],    "Airfoil")
        tabs.addTab(self.panels['wingbox'],    "Wing-Box")
        tabs.addTab(self.panels['planform'],   "Planform")
        tabs.addTab(self.panels['structural'], "Structural")
        layout.addWidget(tabs)

        self.tabs = tabs

    def get_inputs(self):
        idx   = self.tabs.currentIndex()
        name  = list(self.panels.keys())[idx]
        data  = self.panels[name].get_inputs()
        return name, data

class ModeSwitcher(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.stack = QStackedWidget()
        self.flight = FlightMissionInput()
        self.predict = ComponentPredictionPanel()
        self.stack.addWidget(self.flight)
        self.stack.addWidget(self.predict)

        self.switch_btn = QPushButton("Switch to Component Predictions Mode")
        self.switch_btn.clicked.connect(self._toggle)
        layout.addWidget(self.stack)
        layout.addWidget(self.switch_btn)

    def _toggle(self):
        i = 1 - self.stack.currentIndex()
        self.stack.setCurrentIndex(i)
        if i == 0:
            self.switch_btn.setText("Switch to Component Predictions Mode")
        else:
            self.switch_btn.setText("Switch to Flight Mode")

    def get_inputs(self):
        idx = self.stack.currentIndex()
        if idx == 0:
            return 0, self.flight.get_inputs()
        else:
            return 1, self.predict.get_inputs()

class UserInputPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.switcher = ModeSwitcher()
        layout.addWidget(self.switcher)

    def get_inputs(self):
        return self.switcher.get_inputs()
