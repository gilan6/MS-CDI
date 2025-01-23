from dataclasses import dataclass


@dataclass(frozen=True)
class AlgParam:
    """
    iter_num: number of alg. iterations
    alpha: float - controls how much to update in probe weak areas
    gamma: float - update step size
    start_momentum_ind - too soon wil explode, too far will delay
    stop_momentum_ind: float - sometimes momentum is too much
    eta: float - momentum update size
    """

    iter_num: int = int(5e3)
    alpha: float = 0.1
    gamma: float = 0.7
    eta: float = 0.9


@dataclass(frozen=True)
class ScenarioParam:
    # process_num: int = 0  # can be used for multi GPU
    mask_num: int = 2  # number of Masks
    camera_n: int = 512  # TODO: fix duplicate
    snr: float = 30.0
    bit_depth: int = 16
    block_width: int = 0
    obj_num: int = 10  # upto 100


@dataclass(frozen=True)
class MaskParam:
    """
    mask_type: 'speckles'/'phase'/'unfiltered_phase'/'binary'
    filter_shape:  'none'/'square'/'circ',
    filter_inner/outer_limit is the Diameter / side-length
    supp_shape: 'square'/'circ'
    supp_size: diameter / side-length
    supp_out_shape: 'square'/'circ'
    supp_out_size: diameter / side-length
    """

    camera_n: int = 512  # TODO: fix duplicate

    mask_type: str = "binary"
    filter_shape: str = "circ"
    filter_outer_limit: float = camera_n * 0.2
    filter_inner_limit: float = camera_n * 0.1
    supp_shape: str = "square"
    supp_size: int = int(camera_n * 0.35 // 2 * 2)
    supp_out_shape: str = "square"
    supp_out_size: int = int(camera_n * 0.5 // 2 * 2)


@dataclass(frozen=True)
class ObjParam:
    """
    filter_shape:  'none'/'square'/'circ',
    filter_outer_limit is the Diameter / side-length
    support_shape: 'square'/'circ'
    support_size: diameter / side-length
    """

    camera_n: int = 512  # TODO: fix duplicate

    filter_shape: str = "square"
    filter_outer_limit: float = camera_n * 0.8
    supp_shape: str = "square"
    supp_size: int = int(camera_n * 0.45 // 2 * 2)


@dataclass(frozen=True)
class IOParam:
    """
    save / load paths etc.
    """

    save_data_to_file: bool = False
    load_from_file: bool = False
    save_data_path: str = "Results/MsCDI_K_2.pkl"
    save_temp_path: str = "Results/MsCDI_K_2_process"
    plot_file_path: str = "Results/lines.pdf"
    log_deltas: bool = False


@dataclass
class Param:
    scenario: ScenarioParam = ScenarioParam()
    mask: MaskParam = MaskParam(camera_n=ScenarioParam.camera_n)
    obj: ObjParam = ObjParam(camera_n=ScenarioParam.camera_n)
    io: IOParam = IOParam()
    alg: AlgParam = AlgParam()
