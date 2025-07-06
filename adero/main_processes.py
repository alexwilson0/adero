import math
import numpy as np
import tempfile
import json
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtWidgets import QApplication, QFileDialog
import plotly.io as pio
from adero.atmosphere import isa
from adero.visualizer import generate_wing_figure, generate_2d_airfoil_plot
from adero.cst import average_thickness_ratio
from Airfoil_Optimizer.optimiserTemplateFINAL import optimize_airfoil, NN_Call as AF_Call
from Planform_Optimizer.planformOptimiser import optimize_planform, CombinedModel_Call as PF_Call
from Wing_Box_Optimizer.wing_box_optimiser import optimize_structure as optimize_wingbox, CombinedModel_Call as WB_Call
from Structural_Optimizer.structural_optimiser import optimize_section, CombinedModel_Call as ST_Call


class MainProcesses:
    def __init__(self, window):
        self.win = window
        self.last_results = None 

    def log(self, msg):
        self.win.script_reader.append(msg)

    def _set_progress(self, pct):
        self.win.progress.setValue(pct)
        QApplication.processEvents()

    def _parse_weights(self, weights_dict, name):
        """
        Convert empty → 0.0, parse floats, ensure sum == 1.
        Returns parsed dict or None on error.
        """
        parsed = {}
        invalid = []
        for key, val in weights_dict.items():
            s = val.strip()
            if s == "":
                num = 0.0
            else:
                try:
                    num = float(s)
                except ValueError:
                    invalid.append(key)
                    continue
            parsed[key] = num

        if invalid:
            self.log(f"Invalid weight input(s) in {name}: {', '.join(invalid)}\n")
            return None

        total = sum(parsed.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            self.log(f"{name} weights must sum to 1. Current sum: {total}\n")
            return None

        return parsed

    def run(self):
        mode, data = self.win.input_panel.get_inputs()
        if mode == 0:
            mission, aero_w, plan_w, wb_w, sec_w = data
            self._run_flight(mission, aero_w, plan_w, wb_w, sec_w)
        else:
            comp_name, inputs = data
            self._run_prediction(comp_name, inputs)

    def reset(self):
        self.win.progress.setValue(0)
        QApplication.processEvents()
        self.win.script_reader.clear()
        self.win.script_reader.append("System Log:\n")

    def _run_prediction(self, comp, inputs):
        """Dispatch to the correct CombinedModel_Call or NN_Call"""
        try:
            # convert to floats
            vals = [float(v) for v in inputs.values()]
        except ValueError:
            self.log(f"Invalid numeric input in {comp} predictor\n")
            return

        if comp == 'airfoil':
            out = AF_Call(vals)
            names = ['CL0', 'CL_stall', 'dCL_dalpha', 'CD0', 'CD_min', 'CL_mindrag', 'CD_CL0', 'k']
            for n, p in zip(names, out):
                self.log(f"{n}: {p}\n")

        elif comp == 'wingbox':
            out = WB_Call(vals)
            names = ['Tip_Deflection','Max_Stress','Stress_Ratio','Root_Bending_Moment']
            for n, p in zip(names, out):
                self.log(f"{n}: {p}\n")

        elif comp == 'planform':
            out = PF_Call(vals)
            names = ['CL0','CD_CL0','alpha0','a0',
                     'CD_min','CL_min','CD0','k','CM0','dCM_dalpha']
            for n, p in zip(names, out):
                self.log(f"{n}: {p}\n")

        elif comp == 'structural':
            out = ST_Call(vals)
            names = ['Total_Deformation','Shear_Stress_XY','Mass',
                     'Rib_Stress','Spar_Stress','Airfoil_Stress',
                     'Root_Bending_Moment']
            for n, p in zip(names, out):
                self.log(f"{n}: {p}\n")

        else:
            self.log(f"Unknown component: {comp}\n")

    def _run_flight(self, mission, aero_w, plan_w, wb_w, sec_w):
        self.last_results = {
            'mission': mission.copy(),
            'airfoil_weights': aero_w.copy(),
            'planform_weights': plan_w.copy(),
            'wingbox_weights': wb_w.copy(),
            'section_weights': sec_w.copy(),
            'airfoil':   None,
            'wingbox':   None,
            'planform':  None,
            'section':   None,
            'final_wing_params': None      # ← placeholder
        }
        
        STEPS = [
            ("Init",         0),
            ("ISA & Re",    10),
            ("Airfoil",     30),
            ("Wing-Box",    50),
            ("Planform",    70),
            ("Section",     85),
            ("Visualize",  100),
        ]

        # 1) Check for missing fields
        flight_labels = {
            'mtow':             "Maximum Take-Off Weight (kg)",
            'payload_mass':     "Payload Mass (kg)",
            'payload_volume':   "Payload Volume (m^3)",
            'cruise_velocity':  "Cruise Velocity (m/s)",
            'cruise_altitude':  "Cruise Altitude (m)",
            'ceiling_altitude': "Ceiling Altitude (m)",
        }
        missing = [
            label for key, label in flight_labels.items()
            if not mission.get(key, "").strip()
        ]
        if missing:
            self.log(f"Missing input(s): {', '.join(missing)}\n")
            return

        # 2) Check that each is a valid float
        invalid = []
        for key, label in flight_labels.items():
            val = mission.get(key, "").strip()
            try:
                float(val)
            except ValueError:
                invalid.append(label)

        if invalid:
            self.log(f"Invalid inputs(s) in: {', '.join(invalid)}\n")
            return
        
        # 3) Parse and validate each weight group
        aero_w = self._parse_weights(aero_w, "Airfoil optimizer")
        if aero_w is None: return

        plan_w = self._parse_weights(plan_w, "Planform optimizer")
        if plan_w is None: return

        wb_w = self._parse_weights(wb_w, "Wing-box optimizer")
        if wb_w is None: return

        sec_w = self._parse_weights(sec_w, "Section optimizer")
        if sec_w is None: return

        # All inputs present and numeric → proceed
        self.log("Starting wing generation...\n")
        self._set_progress(0)

        try:
            # ISA and Reynolds
            _, _, rho, mu = isa(float(mission['cruise_altitude']))
            Re = rho * float(mission['cruise_velocity']) / mu
            self.log(f"Re: {Re:.5g}\n")
            

            # Airfoil optimization
            self.log(f"OPTIMIZING AEROFOIL...\n")
            self._set_progress(10)
            params, nn_out, perf, hist = optimize_airfoil(Re, aero_w)
            upper = params[1:4].tolist()
            lower = params[4:7].tolist()
            fixed_CST = upper + lower
            self.log(f"Airfoil optimized parameters: {fixed_CST[0]:.5g}, {fixed_CST[1]:.5g}, {fixed_CST[2]:.5g}, {fixed_CST[3]:.5g}, {fixed_CST[4]:.5g}, {fixed_CST[5]:.5g}\n")
            
            # Wing-box optimization
            self.log(f"OPTIMIZING WING-BOX...\n")
            self._set_progress(30)
            tc_avg = average_thickness_ratio(params[1:4].tolist(), params[4:7].tolist())
            mtow = mission['mtow']
            wb_params, wb_outputs = optimize_wingbox(
                fixed_tc_avg=tc_avg,
                fixed_MTOW=mtow,
                weights=wb_w)
            self.log(f"Wing-box optimized parameters: {wb_params[0]:.5g}, {wb_params[1]:.5g}, {wb_params[2]:.5g}\n")
            chord, span, taper_wb = wb_params
            area = span * (chord * (1 + taper_wb)) / 2
            AR = span / chord

            # Planform optimization
            self.log(f"OPTIMIZING PLANFORM...\n")
            self._set_progress(50)
            CL0_des = float(mission['mtow']) / (0.5 * rho * float(mission['cruise_velocity'])**2 * 30)
            pf_vars, pf_outputs = optimize_planform(CL0_des, AR, area,
                                             fixed_CST, plan_w)
            taper = pf_vars[0]
            sweep = pf_vars[1]
            self.log(f"Planform optimized parameters: {pf_vars[0]:5g}, {pf_vars[1]:5g}\n")

            # Structural optimization
            self.log(f"OPTIMIZING STRUCTURE...\n")
            self._set_progress(70)
            tip_chord = chord * taper_wb
            s_params, s_outputs = optimize_section(fixed_root=chord, 
                                                   fixed_tip=tip_chord, 
                                                   fixed_span=span,
                                                   fixed_sweep=sweep,
                                                   weights=sec_w)
            rib_th, spar_th, skin_th, n_ribs = s_params
            n_ribs = int(round(n_ribs))
            self.log(f"Structural optimized parameters: {s_params[0]:.5g}, {s_params[1]:.5g}, {s_params[2]:.5g}, {round(s_params[3])}\n")
            self.log("Optimizations complete!\n")
            self._set_progress(85)
                    # Generate and load figures
            wp = {
                'sweep': sweep,
                'taper': taper_wb,
                'AR': AR,
                'area': area,
                'chord': chord,
                'CST_coeff': {
                    'root': {'upper': params[1:4].tolist(), 'lower': params[4:7].tolist()},
                    'tip':  {'upper': params[1:4].tolist(), 'lower': params[4:7].tolist()}
                },
                'internal': {
                    'rib_thickness':   rib_th,
                    'spar_thickness':  spar_th,
                    'num_ribs':        n_ribs,
                    'num_spars':       2
                }
            }

            # Store for rendering & toggles
            self.win.wing_params = wp

            self.last_results['final_wing_params'] = wp.copy()

            fig3d = generate_wing_figure(wp,
                                         upper_opacity=1,
                                         lower_opacity=1,
                                         show_internal=True)
            fig2d = generate_2d_airfoil_plot(wp)

            for fig, view in ((fig3d, self.win.viewer_3d),
                              (fig2d, self.win.viewer_2d)):
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                tmpf.write(pio.to_html(fig, full_html=True).encode('utf-8'))
                tmpf.close()
                view.load(QUrl.fromLocalFile(tmpf.name))

            self.win._refresh_view()
            self.log("Visualising wing geometry...\n")
            self._set_progress(100)

            span = math.sqrt(wp['AR'] * wp['area'])
            tip_chord = wp['chord'] * wp['taper']

            self.log("\n=== Final Wing Geometry ===")
            self.log(f"Span:              {span:.3f} m")
            self.log(f"Sweep:             {wp['sweep']:.3f} deg")
            self.log(f"Taper Ratio:       {wp['taper']:.3f}")
            self.log(f"Root Chord:        {wp['chord']:.3f} m")
            self.log(f"Tip Chord:         {tip_chord:.3f} m")
            self.log(f"Aspect Ratio:      {wp['AR']:.3f}")
            self.log(f"Wing Area:         {wp['area']:.3f} m²")

            self.log("\n--- Internal Structure ---")
            self.log(f"Rib Thickness:     {wp['internal']['rib_thickness']:.4f} m")
            self.log(f"Spar Thickness:    {wp['internal']['spar_thickness']:.4f} m")
            self.log(f"Number of Ribs:    {wp['internal']['num_ribs']}")
            self.log(f"Number of Spars:   {wp['internal']['num_spars']}")

        except Exception as e:
            self.log(f"Error in flight process: {e}\n")
    
    def save(self):
        mode, _ = self.win.input_panel.get_inputs()
        if mode == 1:
            self.log("Save unavailable in Component Predictions Mode; future version will support it.\n")
            return
        if not self.last_results:
            self.log("No geometry to save.  Please Run first.\n")
            return

        path, _ = QFileDialog.getSaveFileName(
            self.win,
            "Save Wing Generation Results",
            "",
            "Text Files (*.txt)")
        if not path:
            return

        with open(path, 'w') as f:
            f.write("=== Wing Generation Results ===\n\n")

            # — mission —
            f.write("Mission Profile:\n")
            for k,v in self.last_results['mission'].items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

            # — weights —
            # (unchanged)

            # — airfoil —
            # (unchanged)

            # — wingbox, planform, section —
            # (unchanged)

            # — final wing params —
            f.write("Final Wing Parameters (wp):\n")
            f.write(json.dumps(self.last_results['final_wing_params'], indent=2))
            f.write("\n\n")

        self.log(f"Results saved to: {path}\n")
