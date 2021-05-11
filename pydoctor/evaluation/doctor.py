import importlib
import os
import time
import SimpleITK as sitk
import pydicom

from pydoctor.evaluation.environment import env_settings
from pydoctor.utils.visdom import Visdom
import numpy as np

class Doctor:
    """Wraps the doctor for evaluation and running purposes.
     args:
         name: Name of diagnose method.
         parameter_name: Name of parameter file.
         run_id: The run id.
         display_name: Name to be displayed in the result plots.
     """
    def __init__(self, name: str, parameter_name: str,  display_name: str = None):

        self.name = name
        self.parameter_name = parameter_name
        self.display_name = display_name
        env = env_settings()

        self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        doctor_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','doctors',self.name))
        if os.path.isdir(doctor_module_abspath):
            doctor_module = importlib.import_module('pydoctor.doctors.{}'.format(self.name))
            self.doctor_class = doctor_module.get_doctor_class()
        else:
            self.doctor_class = None
        self.visdom = None

    def _init_visdom(self, visdom_info, debug):
        visdom_info = {} if visdom_info is None else visdom_info
        self.pause_mode = False
        self.step = False
        if debug > 0 and visdom_info.get('use_visdom', True):
            try:
                self.visdom = Visdom(debug, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
                                     visdom_info=visdom_info)

                # Show help
                help_text = 'You can pause/unpause the tracker by pressing ''space'' with the ''Tracking'' window ' \
                            'selected. During paused mode, you can track for one frame by pressing the right arrow key.' \
                            'To enable/disable plotting of a data block, tick/untick the corresponding entry in ' \
                            'block list.'
                self.visdom.register(help_text, 'text', 1, 'Help')
            except:
                time.sleep(0.5)
                print('!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
                      '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!')

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True

    def create_doctor(self, params):
        tracker = self.doctor_class(params)
        tracker.visdom = self.visdom
        return tracker

    def run_study(self,std,visdom_info):
        """
        Run doctor on study.
        :param std: Study to run the doctor on .
        :param visualization: Set visualization flag .
        :param debug: Set the debug level .
        :param visdom_info: Visdom info.
        :return:
        """
        params = self.get_parameters()
        self._init_visdom(visdom_info,2)
        doctor = self.create_doctor(params)
        output = self._diagnose_study(doctor,std)
        return output

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('pydoctor.parameter.{}.{}'.format(self.name, self.parameter_name))
        params = param_module.parameters()
        return params

    def _diagnose_study(self, doctor, std):
        frame,pydicom_info_dict = doctor.read_dicom_image(std)
        flag = doctor.initialize()
        if flag:
            annotation = doctor.diagnose(frame,std.index,std.name)
            dict_tmp = {'seriesUid':pydicom_info_dict['seriesUid'],
                        'instanceUid':pydicom_info_dict['instanceUid'],
                        'annotation':annotation}
            output = {'studyUid':pydicom_info_dict['studyUid'],
                      'data':[dict_tmp]}
            return output




    def _read_dicom_image(self, frame_path):
        image = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(frame_path)))
        key_image = np.expand_dims(np.uint8((image - image.min()) / (image.max() - image.min()) * 255.0),0).repeat(3, axis=0)
        pydicom_file = pydicom.read_file(frame_path)
        studyUid = pydicom_file.get(0x0020000D)._value
        seriesUid = pydicom_file.get(0x0020000E)._value
        instanceUid = pydicom_file.get(0x00080018)._value
        uid_info = {'studyUid':str(studyUid),
                    'seriesUid':str(seriesUid),
                    'instanceUid':str(instanceUid)}
        return key_image/255.0, uid_info











