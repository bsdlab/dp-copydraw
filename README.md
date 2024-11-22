# Dareplane CopyDraw module
This repo contains the python implementation of the copy draw recording and analysis pipeline

# Config
Configuration files are stored under `./configs`

# Testing
You should be able to test a single run with This
```bash
python -m copydraw.main 
```
--> If this is not working, something is still odd in the setup

## Calibration
Some calibration of scaling an shifting of the trace might be necessary, depending on the aspect ratio
which is being used for the copydraw psychopy window. This is due to the trace being displayed as a png background image
instead of rendering it as a psychopy trace. We had to resort to this in order to keep up a higher frame rate of recording (usually > 80hz)
It is best to check the calibration by running this:
```bash
python -m copydraw.main --for_calibration=True
```

Should the traces and control points be non overlapping, you have three places to adjust:
1. in `copydraw/elements.py`
```python
    'template': {
        'units': 'norm',
        'pos': (-0.0, 0.01),
        'interpolate': True,
    }
```
The `'pos'` parameter can be used for translations.

2. in `copydraw/copydraw.py`
```python
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
            size=self.block_settings['size'] * 1.67,
        )
```
Here the size parameter is used for an isometric scaling of the picture.

3. in `copydraw/utils/template_tools.py`
```python
def template_to_image(template, fname, path, for_calibration: bool = False,
                      scale: float = 1, **kwargs):

    # if template images dir doesn't exists make it
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    fullpath = path.joinpath(f'{fname}.png')

    if not fullpath.exists():
        # fig = plt.figure(figsize=(16, 10))
        fig = plt.figure(figsize=(16, 9), frameon=False)
```
Here you can adjust the `figsize` according to the target screen.

# Server
Like for other Dareplane modules, one can manually spawn the server via
```bash
python -m api.server
```
and after connecting e.g. via `telnet 127.0.0.1 8080`, try sending a `START_BLOCK` to run a single block of
copydraw.
