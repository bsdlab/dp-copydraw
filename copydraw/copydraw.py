import numpy as np
import scipy.io
import time
import yaml

from copydraw.utils.config_loading import get_nextblock_metadata, load_block_config
from copydraw.elements import create_element
from copydraw.vpport import VPPort
from psychopy import event, clock, visual
from psychopy.tools.monitorunittools import convertToPix  # , posToPix
from pathlib import Path
from copydraw.utils.logging import logger

from copydraw.utils.serialization import serialize_dict_values
from copydraw.utils.template_tools import (
    create_template_order, smooth,
    scale_to_norm_units, template_to_image)


# boxColour = [160,160,180]
# boxHeight = 200
# boxLineWidth = 6
# templateColour = [80,80,150]
# templateThickness = 3
# traceColour = [255,50,50]
# traceThickness = 1
# startTrialBoxColor = [50, 255, 255]
# textColour = [153, 153, 255]
# timeColour = [180, 180, 160]


class CopyDraw:

    def __init__(self,
                 data_dir,
                 script_dir,
                 screen_ix=None,
                 screen_size=(1680, 1050),
                 flip=True,
                 for_calibration: bool = False,   # add calibration dots,
                 serial_nr: str = 'COM4',  # Usb device for the trigger box
                 ):

        self.log = logger

        self.log.info('Initialising..')

        self.vpp = VPPort(serial_nr)

        self.name = 'copyDraw'
        # should be overwritten while running a block
        self.file_prefix = 'STIM_UNKNOWN_'

        # data containers being used within the exec_block part
        self.frame = {}
        self.trial_settings = {}
        self.block_settings = {}
        self.names = {}

        sp = Path(script_dir)
        self.paths = {'script': sp,
                      'configs': sp.joinpath('configs'),
                      'templates': sp.joinpath('copydraw/assets/templates'),
                      'instructions': sp.joinpath('copydraw/assets/instructions/instructions.png'),
                      'data': Path(data_dir),
                      'script_root': sp,
                      'results_path': Path(data_dir).joinpath('copydraw/raw_behavioral')
                      }

        self.win_settings = {
            'screen_size': screen_size,
            'screen_ix': screen_ix
        }

        self.log.debug(f'Paths: {self.paths}')
        self.log.debug(f'Screen: {self.win_settings["screen_ix"]}')

        self.stimuli = {'flip': flip,
                        'the_box': np.array([[285, 392], [1635, 392], [285, 808], [1635, 808],
                                            [288, 392], [288, 808], [1632, 392], [1632, 808]]),
                        }

        self.trials_vec = None
        self.trial_idx = None

        self.block_results = None
        self.trial_results = None

        self.win = None
        self.photobox = None
        self.fixation = None
        self.for_calibration = for_calibration

        # self.init_screen()
        # self.set_block_settings()

        self.log.debug('initialised')

    def init_session(self, session_name=str, ):

        self.names['session'] = session_name
        self.log.info(f'Initialised session: {self.names["session"]}')
        self.paths['session'] = self.paths['results_path']

    def init_screen(self, win_color: tuple[int, int, int] = (-1, -1, -1)):
        self.win = visual.Window(
            #fullscr=True if self.win_settings['screen_size'] is None else False,
            fullscr=False,
            size=(self.win_settings['screen_size']
                  if self.win_settings['screen_size'] is not None else (800, 600)),
            units='norm',
            screen=self.win_settings['screen_ix'],
            color=win_color
        )
        # needed for setting initial cursor position
        self.win_settings['aspect_ratio'] = self.win.size[0] / self.win.size[1]
        self.photobox = self.get_photobox()

    def set_block_settings(self,
                           n_trials=1,
                           letter_time=2.2,
                           finish_when_raised=False,
                           n_letters=3,
                           stim_size=35,  # this is the size of the templates used
                           size=.5,  # scaling factor applied to the template -> by default smaller than full screen!
                           interp=True,
                           shuffle=True,
                           # block_nr: None | int = None
                           block_nr: int = None
                           ):

        # supplied in the block_config.yaml
        self.block_settings['n_trials'] = n_trials
        self.block_settings['letter_time'] = letter_time
        self.block_settings['n_letters'] = n_letters
        self.block_settings['finish_when_raised'] = finish_when_raised
        self.block_settings['stim_size'] = stim_size
        self.block_settings['size'] = size

        # currently supplied only in this function
        self.block_settings['interp'] = interp
        self.block_settings['shuffle'] = shuffle

        if block_nr is None:
            block_idx, block_name = get_nextblock_metadata(
                self.paths['results_path'])
            self.block_settings['block_idx'] = block_idx
        else:
            self.block_settings['block_idx'] = block_nr

    def init_block(self):
        self.init_screen()
        block_nr = self.block_settings['block_idx']

        self.block_settings['block_name'] = \
            f"block_{block_nr:02}" if block_nr is not None else f'BLOCK_{time.strftime("%Y-%m-%d")}'

        # Check if recordings for this block nr already exist
        outfolder =self.paths['session'].joinpath(self.block_settings['block_name'])
        if outfolder.exists():
            self.block_settings['block_name'] += f'_{time.strftime("%Y%m%d%H%M%S")}'
            outfolder =self.paths['session'].joinpath(self.block_settings['block_name'])

        self.log.info(
            f'Initialised block: {self.block_settings}\n{self.win_settings}')

        self.load_stimuli(self.paths['templates'],
                          short=True if self.block_settings['n_letters'] == 2 else False,
                          size=self.block_settings['stim_size'])

        # folders for saving
        self.paths['block'] = outfolder
        self.paths['block'].mkdir(parents=True, exist_ok=True)

        # create trials vector
        self.trials_vec = {
            'lengths': [],  # n_letters
            'id': []  # index
        }

        # trial index
        self.trial_idx = 1

        # create template order
        self.stimuli['order'] = create_template_order(self.stimuli,
                                                      self.block_settings)

    def exec_block(self,
                   # block_nr: None | int = None,
                   # stim: None | str = None,
                   # block_config: None | dict = None
                   block_nr: int = None,
                   stim: str = None,
                   block_config: dict = None,
                   data_root: str | None = None
                   ) -> int:   # TODO: Make this return a subprocess which is running the whole psychcopy stuff
        """ Will call init_block(**cfg) before calling exec trial n_trials
        times, also calling save_trial for each. Trials vector and block
         settings saved at the end.

         Note: Processing a file_name will not be implemented to keep the nomenclature consistent. Only pass a prefix
         and a block_nr

         """
        if data_root:
            self.paths['session'] = Path(data_root)

        self.file_prefix = f'STIM_{stim}_' if stim is not None else 'STIM_UNKNOWN_'
        logger.debug(f">> Using {self.file_prefix=}")
        self.block_settings = {}

        if block_config is None:
            block_config = load_block_config(script_root=self.paths['script_root'])

        # function needed to set default values, in case
        # they are missing in the block config
        self.set_block_settings(**block_config, block_nr=block_nr)
        self.init_block()

        self.log.info(f'executing block {self.block_settings["block_idx"]}')
        self.block_results = {}

        for stimuli_idx in range(self.block_settings['n_trials']):
            self.exec_trial(stimuli_idx)
            self.save_trial()
            self.trial_idx += 1
            self.block_results[stimuli_idx] = self.trial_results

        self.save_block_settings()

        self.block_settings = {}
        self.log.info('Block setting resetted to {} - block done')

        # Clean-up closing the window
        self.finish_block()

        return 0

    def load_stimuli(self, path, short=True, size=35):
        self.stimuli['fname'] = f'Size_{"short_" if short else ""}{size}.mat'
        self.paths['stimuli'] = Path(path, self.stimuli['fname'])
        self.log.info(f'loading stimuli: {self.paths["stimuli"]}')
        assert self.paths['stimuli'].exists(
        ), f"Stimuli data not found: {self.paths['stimuli']}"

        self.stimuli['file'] = scipy.io.loadmat(self.paths['stimuli'])
        self.log.info('loaded mat file')
        self.stimuli['templates'] = self.stimuli['file']['new_shapes'][0]

        self.log.info('stimuli loaded')

        self.stimuli['n_templates'] = self.stimuli['templates'].shape[0]

    def scale_stimuli(self):
        """ Scaling transforms the position to normalized 2d coordinates (-.5, .5) x (-.5, 5)"""
        self.log.info('Scaling data stim...')
        self.stimuli['current_stim'], self.stimuli['scaling_matrix'] = \
            scale_to_norm_units(self.stimuli['current_stim'])
        self.stimuli['the_box'], _ = \
            scale_to_norm_units(self.stimuli['the_box'],
                                scaling_matrix=self.stimuli['scaling_matrix'])

        # reorder
        new_box_array = np.zeros([4, 2])
        new_box_array[0:2] = self.stimuli['the_box'][0:2]
        new_box_array[2] = self.stimuli['the_box'][3]
        new_box_array[3] = self.stimuli['the_box'][2]

        self.stimuli['the_box'] = new_box_array

        self.log.info('scaled')
        if self.stimuli['flip']:
            self.log.info('Flipping data...')
            self.stimuli['current_stim'] = \
                np.matmul(self.stimuli['current_stim'],
                          np.array([[1, 0], [0, -1]]))
            self.stimuli['the_box'] = np.matmul(self.stimuli['the_box'],
                                                np.array([[1, 0], [0, -1]]))

    def get_stimuli(self, stimuli_idx: int, scale=True):
        # this func is a little long, how can it be broken up?

        self.stimuli['current_stim'] = self.stimuli['templates'][stimuli_idx].astype(
            float)
        self.trials_vec['id'].append(stimuli_idx)
        self.trials_vec['lengths'].append(self.block_settings['n_letters'])
        if scale:
            self.scale_stimuli()

        if self.block_settings['interp']:
            self.log.info('Smoothing data...')
            self.stimuli['current_stim'] = \
                smooth(self.stimuli['current_stim'], return_df=False)

        self.trial_settings['cursor_start_pos'] = self.stimuli['current_stim'][0]
        return self.stimuli['current_stim']

    def save_trial(self):
        # rudimentary atm, can ble cleaned up, flattened a little maybe

        fname = f'{self.file_prefix}copyDraw_block{self.block_settings["block_idx"]}_trial{self.trial_idx}.yaml'
        fpath = self.paths['block'].joinpath(fname)

        with open(fpath, 'w') as f:
            logger.info(f"Writing CopyDraw behavioral data to: {fpath=}")
            yaml.safe_dump(serialize_dict_values(self.trial_results), f)

        self.log.info(f'Saved trial: {fpath}')

    def save_block_settings(self):
        fname = f'copyDraw_block{self.block_settings["block_idx"]}_settings.yaml'
        fpath = self.paths['block'].joinpath(fname)

        self.block_settings['trials_vec'] = self.trials_vec
        self.block_settings['templates'] = self.stimuli['templates']

        with open(fpath, 'w') as f:
            yaml.safe_dump(serialize_dict_values(self.block_settings), f)

        self.log.info(f'Saved block settings: {fpath}')

    # draw order is based on .draw() call order, consider using an ordered dict?
    # MD: Ordered dict would be a good idea
    def draw_and_flip(self, exclude=[]):
        """ Draws every element in the frame elements dict, excluding those
         passed in via exclude. """
        for element_name, element in self.frame['elements'].items():
            if element_name in exclude:
                continue
            else:
                element.draw()
        self.win.flip()

    # MD Maybe we could make the frame a class for itself?
    def create_frame(self, stimuli_idx, scale=True, for_calibration: bool = False):

        self.frame['elements'] = {}
        self.frame['elements']['template'] = create_element(
            'template',
            win=self.win,
            image=template_to_image(
                self.get_stimuli(stimuli_idx, scale=scale),
                f'{self.stimuli["fname"][:-4]}_{stimuli_idx}',
                self.paths['data'].joinpath('template_images'),
                linewidth=15,
                for_calibration=for_calibration,
            ),
            # scaling (trial and error using for_calibration==True - somewhere between 16/9 = 1.7777 and 16/10 = 1.6)
            size=self.block_settings['size'] * 1.67,
        )

        template_scaling = 1.5
        if for_calibration:
            self.add_calibration_frame_elements(
                stimuli_idx=stimuli_idx, scale=scale,
                size=template_scaling * self.block_settings['size']
            )

        self.frame['elements']['template'].setOpacity(0.8)
        self.frame['elements']['the_box'] = create_element('the_box',
                                                           win=self.win,
                                                           vertices=self.stimuli['the_box']
                                                           )

        self.frame['elements']['trial_number'] = create_element(
            'trial_number',
            win=self.win,
            text=f'Trial {self.trial_idx}/{self.block_settings["n_trials"]}',
        )

        self.log.debug(self.trial_settings)
        self.trial_settings['template_scaling'] = template_scaling
        self.frame['elements']['cursor'] = create_element(
            'cursor',
            win=self.win,
            pos=convertToPix(
                # so the 1.5 comes from how much the original template had to be scaled by after being converted
                # to norm units. the 1.6.. in the template section is how much the resulting image of the template had
                # to be scaled by in order to match up with the original template when drawn on top of each other
                self.trial_settings['cursor_start_pos'] * \
                template_scaling * self.block_settings['size'],
                (0, 0),
                'norm',
                self.win
            )
        )

        max_trace_len = 10000  # Should be more points than would ever be drawn
        self.frame['trace_vertices'] = np.zeros([max_trace_len, 2])
        self.frame['trace_vertices'][0] = \
            convertToPix(self.trial_settings['cursor_start_pos'] * template_scaling * self.block_settings['size'],
                         (0, 0),
                         'norm',
                         self.win)

        self.frame['elements']['trace1'] = create_element(
            # we will dynamically create more -> draw interupted lines
            'trace',
            win=self.win,
            vertices=np.zeros(shape=(1, 2))
        )
        self.frame['traces'].append('trace1')

        self.frame['elements']['instructions'] = create_element(
            'instructions',
            win=self.win,
            image=self.paths['instructions']
        )

        # start_point
        start_point_size = 0.05
        self.frame['elements']['start_point'] = create_element(
            'start_point',
            win=self.win,
            size=(start_point_size, start_point_size *
                  self.win_settings['aspect_ratio'])
        )

        self.frame['elements']['time_bar'] = create_element(
            'time_bar',
            win=self.win
        )

    def add_calibration_frame_elements(self, stimuli_idx: int, scale: bool, size: float = 1.5):
        vert = self.get_stimuli(stimuli_idx, scale=scale)
        self.frame['elements']['old_template'] = visual.shape.ShapeStim(win=self.win,
                                                                        vertices=vert,
                                                                        lineWidth=100,
                                                                        closeShape=False,
                                                                        interpolate=True,
                                                                        ori=0,
                                                                        pos=(
                                                                            0, 0),
                                                                        size=size,
                                                                        units='norm',
                                                                        fillColor='blue',
                                                                        lineColor='blue',
                                                                        # windingRule=True,
                                                                        opacity=0.5
                                                                        )

        # add a grid
        print(
            f"{vert[:, 0].min()=}, {vert[:, 0].max()=}, {vert[:, 1].min()=}, {vert[:, 1].max()=}, ")
        scale = 1.5

        xx, yy = np.meshgrid(
            np.linspace(vert[:, 0].min() * scale,
                        vert[:, 0].max() * scale, 10),
            np.linspace(vert[:, 1].min() * scale,
                        vert[:, 1].max() * scale, 10),
        )
        for i, (x, y) in enumerate(zip(xx.flatten(), yy.flatten())):
            self.frame['elements'][f'psych_grid_{i}'] = visual.circle.Circle(self.win, radius=(.01 / 2, 0.017 / 2),
                                                                             pos=(
                                                                                 x, y), fillColor=None,
                                                                             lineColor='red'
                                                                             )

        self.frame['elements']['topleft'] = visual.circle.Circle(
            self.win, radius=(.02 / 2, 0.034 / 2),
            pos=(vert[:, 0].min() * scale, vert[:, 1].max() * scale), fillColor=None, lineColor='red')
        self.frame['elements']['topright'] = visual.circle.Circle(
            self.win, radius=(.02 / 2, 0.034 / 2),
            pos=(vert[:, 0].max() * scale, vert[:, 1].max() * scale), fillColor=None, lineColor='red')
        self.frame['elements']['botleft'] = visual.circle.Circle(
            self.win, radius=(.02 / 2, 0.034 / 2),
            pos=(vert[:, 0].min() * scale, vert[:, 1].min() * scale), fillColor=None, lineColor='red')
        self.frame['elements']['botright'] = visual.circle.Circle(
            self.win, radius=(.02 / 2, 0.034 / 2),
            pos=(vert[:, 0].max() * scale, vert[:, 1].min() * scale), fillColor=None, lineColor='red')
        self.frame['elements']['00'] = visual.circle.Circle(
            self.win, radius=(.02 / 2, 0.034 / 2),
            pos=(0, 0), fillColor=None, lineColor='red')

    # should stimuli_idx just be called trial_idx?
    # currently they are initialised differently (0 and 1)
    # MD: lets check this, could be that we have 6 stimuli (6 unique traces), but 12 trials
    def exec_trial(self, stimuli_idx, scale=True):
        """ Top level method that executes a single trial """

        # track if mouse is currently lifted
        self.frame['lifted'] = True
        # frame index of the last started trace
        self.frame['start_frame_idx'] = 0
        # list to record multiple trace object names-> drawing interupted lines
        self.frame['traces'] = []
        self.log.info(f'Executing trial with stim idx {stimuli_idx}')

        self.trial_settings['trial_duration'] = self.block_settings['n_letters'] * \
            self.block_settings['letter_time']

        # initialise the frame
        self.create_frame(self.stimuli['order'][stimuli_idx], scale=scale,
                          for_calibration=self.for_calibration)

        # draw first frame
        self.draw_and_flip(exclude=['time_bar', 'trace1', 'the_box'])

        # time_bar
        time_bar_x = self.frame['elements']['time_bar'].size[0]

        # main bit
        self.frame['idx'] = 0  # refers only to frames during drawing trace
        ptt, start_t_stamp, cursor_t = self._run_trial_main_loop(clock,
                                                                 time_bar_x, stimuli_idx)

        cursor_t = cursor_t[:self.frame['idx']+1]  # truncate

        # trial time is how long they were drawing for,
        # ie time elapsed during drawing
        trial_time = self.trial_settings['trial_duration'] - cursor_t[-1]
        self.log.info(f'Trial lasted {trial_time} seconds')
        self.log.info(f'Drew {self.frame["idx"]} frames')
        self.log.info(
            f'Recording rate was {len(cursor_t) / trial_time} points per second')

        # For interupted traces, we now have to concatenate
        # General TODO: --> how to deal with the jumps in terms of dtw...
        trace_let = np.concatenate([self.frame['elements'][tr_n].vertices.copy()
                                    for tr_n in self.frame['traces']])
        traces_pix = [self.frame['elements'][tr_n].verticesPix.copy()
                      for tr_n in self.frame['traces']]
        trace_let_pix = np.concatenate(traces_pix)

        self._create_trial_res(trace_let, trial_time, ptt, start_t_stamp,
                               trace_let_pix, scale, cursor_t, traces_pix)

    def exit(self):
        self.log.info('Exiting')
        self.finish_block()  # base class method

    def finish_block(self):
        # q = input('Press any key to close trace')
        self.win.close()
        # Clean up the files
        for f in Path('./data/VPtest/template_images').rglob('*.png'):
            f.unlink()

    def send_marker(self, val):
        if type(val) == int and val < 256:
            if self.vpp:
                self.vpp.write([val])
        else:
            raise ValueError(
                "Please provide an int value < 256 to be written as a marker")
        logger.info('marker-%d' % val)

    def _run_trial_main_loop(self, clock, time_bar_x, stim_idx: int):
        """ To run the main drawing loop """
        started_drawing = False
        cursor_t = np.zeros([10000])  # for recording times with cursor pos
        ptt = None
        start_t_stamp = None
        mouse = event.Mouse(win=self.win)

        while True:
            self.photobox['black'].draw()
            self.draw_and_flip(exclude=['trace', 'the_box'])

            # Cursor hit the cyan square
            if mouse.isPressedIn(self.frame['elements']['start_point']):
                self.send_marker(50)
                self.photobox['white'].draw()
                self.log.debug('Mouse pressed in startpoint')
                self.frame['elements']['start_point'].fillColor = [-1, 1, 1]
                tic = clock.getTime()
                self.draw_and_flip(
                    exclude=['trace', 'instructions', 'the_box'])
                break

        while True:
            # drawing has begun
            if mouse.isPressedIn(self.frame['elements']['cursor']):
                self.photobox['black'].draw()
                self.send_marker(10 + stim_idx)
                self.log.debug('Mouse started drawing with cursor')
                started_drawing = True
                self.frame['lifted'] = False

                # save start time
                start_t_stamp = clock.getTime()

                # calc pre trial time
                ptt = start_t_stamp - tic

            # shrink time bar, draw trace, once drawing has started
            if started_drawing:
                self.log.info('STARTED DRAWING')
                trial_timer = clock.CountdownTimer(
                    self.trial_settings['trial_duration'])
                self._exec_drawing(trial_timer, mouse, time_bar_x, cursor_t)
                break

        self.photobox['white'].draw()
        self.send_marker(110 + stim_idx)

        return ptt, start_t_stamp, cursor_t

    def _adjust_time_bar(self, ratio, time_bar_x):
        """ Method for adjusting the size of time bar. Wrote
         mainly to aid in profiling. """
        new_size = [time_bar_x * ratio,  # change the x value
                    self.frame['elements']['time_bar'].size[1]]
        new_pos = [(-time_bar_x * ratio / 2) + time_bar_x / 2,
                   self.frame['elements']['time_bar'].pos[1]]

        self.frame['elements']['time_bar'].setSize(new_size)
        self.frame['elements']['time_bar'].setPos(new_pos)

    def _move_cursor(self, mouse, t_remain, cursor_t):
        """ Method for adjusting the position of the cursor and drawing the
        trace. Wrote mainly to aid profiling."""

        new_trace = False
        # Get new position from mouse
        if mouse.getPressed()[0]:
            if self.frame['lifted']:
                new_trace = True
                self.frame['start_frame_idx'] = self.frame['idx'] + 1
            self.frame['lifted'] = False
            new_pos = convertToPix(mouse.getPos(), (0, 0), units=mouse.units,
                                   win=self.win)
        else:

            self.frame['lifted'] = True
            new_pos = self.frame['trace_vertices'][self.frame['idx']]

        # Record time at which that happened
        # cursor_t.append(t_remain)
        cursor_t[self.frame['idx']] = t_remain

        # Move cursor to that position and save
        self.frame['elements']['cursor'].pos = new_pos

        # For drawing trace
        self._draw_trace(new_pos, new_trace=new_trace)

    def _draw_trace(self, new_pos, new_trace=False):
        """ Method that controls trace drawing (doesn't actually draw it just
        controls the vertices). - for profiling. """
        # Draw trace, or rather, add new_pos to the trace
        self.frame['trace_vertices'][self.frame['idx']+1] = new_pos

        if new_trace:
            self.log.info('New trace')
            tr_i = int(self.frame['traces'][-1].replace('trace', '')) + 1
            tr_i_n = 'trace' + str(tr_i)
            self.frame['elements'][tr_i_n] = create_element(
                'trace',
                win=self.win,
                vertices=np.zeros(shape=(1, 2))
            )
            self.frame['traces'].append(tr_i_n)

        # set the trace
        self.frame['elements'][self.frame['traces'][-1]].vertices =\
            self.frame['trace_vertices'][self.frame['start_frame_idx']                                         :self.frame['idx']+1]

    def _exec_drawing(self, trial_timer, mouse, time_bar_x, cursor_t):
        """ All the drawing stuff goes in this """
        while trial_timer.getTime() > 0:

            # get remaining time
            t_remain = trial_timer.getTime()
            ratio = t_remain / self.trial_settings['trial_duration']

            # adjust time_bar size and position
            if self.frame['idx'] % 2 == 0:  # every other frame
                self._adjust_time_bar(ratio, time_bar_x)

            # update cursor position
            self._move_cursor(mouse, trial_timer.getTime(), cursor_t)

            # only draw every xth frame
            if self.frame['idx'] % 2 == 0:
                self.draw_and_flip(exclude=['instructions', 'the_box'])

            if (not mouse.getPressed()[0] and
                    self.block_settings['finish_when_raised']):
                self.log.info('mouse raised - ending trial')
                break
            self.frame['idx'] += 1

    def _create_trial_res(self, trace_let, trial_time, ptt, start_t_stamp,
                          trace_let_pix, scale, cursor_t, traces_pix):
        """ Creates the results dict that contains basic/essential trial info
        to be saved. """

        self.log.debug('Creating trial results')

        # original data + metadata
        self.trial_results = {'trace_let': trace_let,
                              'trial_time': trial_time,
                              'ix_block': self.block_settings["block_idx"],
                              'ix_trial': self.trial_idx,
                              'ptt': ptt,
                              'start_t_stamp': start_t_stamp}

        # new/extra metadata
        if scale:
            # this is for pix to relative coords
            self.trial_results['scaling_matrix'] = self.stimuli['scaling_matrix']

        # this is the isometric stretch which was applied to the template onto the screen
        # scaling the template_pix with this will result in the actual screens coordinates which are also tracked
        # in traces_pix
        self.trial_results['template_scaling'] = self.trial_settings['template_scaling']

        self.trial_results['traces_pix'] = traces_pix
        self.trial_results['n_traces'] = len(self.frame['traces'])
        self.trial_results['trial_duration_config'] = self.trial_settings['trial_duration']
        self.trial_results['flip'] = self.stimuli['flip']
        self.trial_results['theBox'] = self.frame['elements'][
            'the_box'].vertices.copy()

        self.trial_results['theBoxPix'] = self.frame['elements'][
            'the_box'].verticesPix

        self.trial_results['cursor_t'] = cursor_t

        if (trace_let != trace_let_pix).any():
            self.trial_results['pos_t_pix'] = trace_let_pix

        # in matlab i think this is theRect
        self.trial_results['winSize'] = self.win.size

        self.trial_results['template'] = self.stimuli['current_stim']
        stim_size = self.frame['elements']['template'].size
        stim_pos = self.frame['elements']['template'].pos
        self.trial_results['template_pix'] = convertToPix(self.stimuli['current_stim'],
                                                          units='norm',
                                                          pos=stim_pos,
                                                          win=self.win)
        self.trial_results['template_size'] = stim_size
        self.trial_results['template_pos'] = stim_pos
        # self.trial_results['templatePix'] = self.frame['elements'][
        #     'template'].verticesPix
        # do i need to add theWord? maybe - is the index enough?

    def get_fixation(self):
        return visual.Circle(self.win, radius=5, color='red', interpolate=True, units='pix')

    def get_photobox(self):
        rect_white = visual.Rect(
            win=self.win, units="pix", width=40, height=40,
            fillColor=[1, 1, 1]
        )
        rect_black = visual.Rect(
            win=self.win, units="pix", width=40, height=40,
            fillColor=[-1, -1, -1]
        )

        rect_white.pos = (10, 500)  # (10, 510)  # (0, 520)
        # x and y coordinates relative to the screen center
        rect_black.pos = (10, 500)  # (10, 510)  # (0, 520)
        return {'white': rect_white, 'black': rect_black}
