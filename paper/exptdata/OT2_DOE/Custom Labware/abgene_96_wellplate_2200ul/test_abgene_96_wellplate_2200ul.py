import json
from opentrons import protocol_api, types

CALIBRATION_CROSS_COORDS = {
    '1': {
        'x': 12.13,
        'y': 9.0,
        'z': 0.0
    },
    '3': {
        'x': 380.87,
        'y': 9.0,
        'z': 0.0
    },
    '7': {
        'x': 12.13,
        'y': 258.0,
        'z': 0.0
    }
}
CALIBRATION_CROSS_SLOTS = ['1', '3', '7']
TEST_LABWARE_SLOT = '2'

RATE = 0.25  # % of default speeds
SLOWER_RATE = 0.1

PIPETTE_MOUNT = 'right'
PIPETTE_NAME = 'p1000_single'

TIPRACK_SLOT = '5'
TIPRACK_LOADNAME = 'opentrons_96_tiprack_1000ul'

LABWARE_DEF_JSON = """{"ordering":[["A1","B1","C1","D1","E1","F1","G1","H1"],["A2","B2","C2","D2","E2","F2","G2","H2"],["A3","B3","C3","D3","E3","F3","G3","H3"],["A4","B4","C4","D4","E4","F4","G4","H4"],["A5","B5","C5","D5","E5","F5","G5","H5"],["A6","B6","C6","D6","E6","F6","G6","H6"],["A7","B7","C7","D7","E7","F7","G7","H7"],["A8","B8","C8","D8","E8","F8","G8","H8"],["A9","B9","C9","D9","E9","F9","G9","H9"],["A10","B10","C10","D10","E10","F10","G10","H10"],["A11","B11","C11","D11","E11","F11","G11","H11"],["A12","B12","C12","D12","E12","F12","G12","H12"]],"brand":{"brand":"Abgene","brandId":["AB0788"]},"metadata":{"displayName":"Abgene 96 Well Plate 2200 µL","displayCategory":"wellPlate","displayVolumeUnits":"µL","tags":[]},"dimensions":{"xDimension":127.76,"yDimension":85.48,"zDimension":42.5},"wells":{"A1":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":14.38,"y":74.24,"z":3.5},"B1":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":14.38,"y":65.24,"z":3.5},"C1":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":14.38,"y":56.24,"z":3.5},"D1":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":14.38,"y":47.24,"z":3.5},"E1":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":14.38,"y":38.24,"z":3.5},"F1":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":14.38,"y":29.24,"z":3.5},"G1":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":14.38,"y":20.24,"z":3.5},"H1":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":14.38,"y":11.24,"z":3.5},"A2":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":23.38,"y":74.24,"z":3.5},"B2":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":23.38,"y":65.24,"z":3.5},"C2":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":23.38,"y":56.24,"z":3.5},"D2":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":23.38,"y":47.24,"z":3.5},"E2":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":23.38,"y":38.24,"z":3.5},"F2":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":23.38,"y":29.24,"z":3.5},"G2":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":23.38,"y":20.24,"z":3.5},"H2":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":23.38,"y":11.24,"z":3.5},"A3":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":32.38,"y":74.24,"z":3.5},"B3":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":32.38,"y":65.24,"z":3.5},"C3":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":32.38,"y":56.24,"z":3.5},"D3":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":32.38,"y":47.24,"z":3.5},"E3":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":32.38,"y":38.24,"z":3.5},"F3":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":32.38,"y":29.24,"z":3.5},"G3":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":32.38,"y":20.24,"z":3.5},"H3":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":32.38,"y":11.24,"z":3.5},"A4":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":41.38,"y":74.24,"z":3.5},"B4":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":41.38,"y":65.24,"z":3.5},"C4":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":41.38,"y":56.24,"z":3.5},"D4":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":41.38,"y":47.24,"z":3.5},"E4":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":41.38,"y":38.24,"z":3.5},"F4":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":41.38,"y":29.24,"z":3.5},"G4":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":41.38,"y":20.24,"z":3.5},"H4":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":41.38,"y":11.24,"z":3.5},"A5":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":50.38,"y":74.24,"z":3.5},"B5":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":50.38,"y":65.24,"z":3.5},"C5":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":50.38,"y":56.24,"z":3.5},"D5":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":50.38,"y":47.24,"z":3.5},"E5":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":50.38,"y":38.24,"z":3.5},"F5":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":50.38,"y":29.24,"z":3.5},"G5":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":50.38,"y":20.24,"z":3.5},"H5":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":50.38,"y":11.24,"z":3.5},"A6":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":59.38,"y":74.24,"z":3.5},"B6":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":59.38,"y":65.24,"z":3.5},"C6":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":59.38,"y":56.24,"z":3.5},"D6":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":59.38,"y":47.24,"z":3.5},"E6":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":59.38,"y":38.24,"z":3.5},"F6":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":59.38,"y":29.24,"z":3.5},"G6":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":59.38,"y":20.24,"z":3.5},"H6":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":59.38,"y":11.24,"z":3.5},"A7":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":68.38,"y":74.24,"z":3.5},"B7":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":68.38,"y":65.24,"z":3.5},"C7":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":68.38,"y":56.24,"z":3.5},"D7":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":68.38,"y":47.24,"z":3.5},"E7":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":68.38,"y":38.24,"z":3.5},"F7":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":68.38,"y":29.24,"z":3.5},"G7":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":68.38,"y":20.24,"z":3.5},"H7":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":68.38,"y":11.24,"z":3.5},"A8":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":77.38,"y":74.24,"z":3.5},"B8":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":77.38,"y":65.24,"z":3.5},"C8":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":77.38,"y":56.24,"z":3.5},"D8":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":77.38,"y":47.24,"z":3.5},"E8":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":77.38,"y":38.24,"z":3.5},"F8":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":77.38,"y":29.24,"z":3.5},"G8":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":77.38,"y":20.24,"z":3.5},"H8":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":77.38,"y":11.24,"z":3.5},"A9":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":86.38,"y":74.24,"z":3.5},"B9":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":86.38,"y":65.24,"z":3.5},"C9":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":86.38,"y":56.24,"z":3.5},"D9":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":86.38,"y":47.24,"z":3.5},"E9":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":86.38,"y":38.24,"z":3.5},"F9":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":86.38,"y":29.24,"z":3.5},"G9":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":86.38,"y":20.24,"z":3.5},"H9":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":86.38,"y":11.24,"z":3.5},"A10":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":95.38,"y":74.24,"z":3.5},"B10":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":95.38,"y":65.24,"z":3.5},"C10":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":95.38,"y":56.24,"z":3.5},"D10":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":95.38,"y":47.24,"z":3.5},"E10":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":95.38,"y":38.24,"z":3.5},"F10":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":95.38,"y":29.24,"z":3.5},"G10":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":95.38,"y":20.24,"z":3.5},"H10":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":95.38,"y":11.24,"z":3.5},"A11":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":104.38,"y":74.24,"z":3.5},"B11":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":104.38,"y":65.24,"z":3.5},"C11":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":104.38,"y":56.24,"z":3.5},"D11":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":104.38,"y":47.24,"z":3.5},"E11":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":104.38,"y":38.24,"z":3.5},"F11":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":104.38,"y":29.24,"z":3.5},"G11":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":104.38,"y":20.24,"z":3.5},"H11":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":104.38,"y":11.24,"z":3.5},"A12":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":113.38,"y":74.24,"z":3.5},"B12":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":113.38,"y":65.24,"z":3.5},"C12":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":113.38,"y":56.24,"z":3.5},"D12":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":113.38,"y":47.24,"z":3.5},"E12":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":113.38,"y":38.24,"z":3.5},"F12":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":113.38,"y":29.24,"z":3.5},"G12":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":113.38,"y":20.24,"z":3.5},"H12":{"depth":39,"totalLiquidVolume":2200,"shape":"rectangular","xDimension":8.4,"yDimension":8.4,"x":113.38,"y":11.24,"z":3.5}},"groups":[{"metadata":{"wellBottomShape":"flat"},"wells":["A1","B1","C1","D1","E1","F1","G1","H1","A2","B2","C2","D2","E2","F2","G2","H2","A3","B3","C3","D3","E3","F3","G3","H3","A4","B4","C4","D4","E4","F4","G4","H4","A5","B5","C5","D5","E5","F5","G5","H5","A6","B6","C6","D6","E6","F6","G6","H6","A7","B7","C7","D7","E7","F7","G7","H7","A8","B8","C8","D8","E8","F8","G8","H8","A9","B9","C9","D9","E9","F9","G9","H9","A10","B10","C10","D10","E10","F10","G10","H10","A11","B11","C11","D11","E11","F11","G11","H11","A12","B12","C12","D12","E12","F12","G12","H12"]}],"parameters":{"format":"irregular","quirks":[],"isTiprack":false,"isMagneticModuleCompatible":false,"loadName":"abgene_96_wellplate_2200ul"},"namespace":"custom_beta","version":1,"schemaVersion":2,"cornerOffsetFromSlot":{"x":0,"y":0,"z":0}}"""
LABWARE_DEF = json.loads(LABWARE_DEF_JSON)
LABWARE_LABEL = LABWARE_DEF.get('metadata', {}).get(
    'displayName', 'test labware')

metadata = {'apiLevel': '2.0'}


def uniq(l):
    res = []
    for i in l:
        if i not in res:
            res.append(i)
    return res


def run(protocol: protocol_api.ProtocolContext):
    tiprack = protocol.load_labware(TIPRACK_LOADNAME, TIPRACK_SLOT)
    pipette = protocol.load_instrument(
        PIPETTE_NAME, PIPETTE_MOUNT, tip_racks=[tiprack])

    test_labware = protocol.load_labware_from_definition(
        LABWARE_DEF,
        TEST_LABWARE_SLOT,
        LABWARE_LABEL,
    )

    num_cols = len(LABWARE_DEF.get('ordering', [[]]))
    num_rows = len(LABWARE_DEF.get('ordering', [[]])[0])
    well_locs = uniq([
        'A1',
        '{}{}'.format(chr(ord('A') + num_rows - 1), str(num_cols))])

    pipette.pick_up_tip()

    def set_speeds(rate):
        protocol.max_speeds.update({
            'X': (600 * rate),
            'Y': (400 * rate),
            'Z': (125 * rate),
            'A': (125 * rate),
        })

        speed_max = max(protocol.max_speeds.values())

        for instr in protocol.loaded_instruments.values():
            instr.default_speed = speed_max

    set_speeds(RATE)

    for slot in CALIBRATION_CROSS_SLOTS:
        coordinate = CALIBRATION_CROSS_COORDS[slot]
        location = types.Location(point=types.Point(**coordinate),
                                  labware=None)
        pipette.move_to(location)
        protocol.pause(
            f"Confirm {PIPETTE_MOUNT} pipette is at slot {slot} calibration cross")

    pipette.home()
    protocol.pause(f"Place your labware in Slot {TEST_LABWARE_SLOT}")

    for well_loc in well_locs:
        well = test_labware.well(well_loc)
        all_4_edges = [
            [well._from_center_cartesian(x=-1, y=0, z=1), 'left'],
            [well._from_center_cartesian(x=1, y=0, z=1), 'right'],
            [well._from_center_cartesian(x=0, y=-1, z=1), 'front'],
            [well._from_center_cartesian(x=0, y=1, z=1), 'back']
        ]

        set_speeds(RATE)
        pipette.move_to(well.top())
        protocol.pause("Moved to the top of the well")

        for edge_pos, edge_name in all_4_edges:
            set_speeds(SLOWER_RATE)
            edge_location = types.Location(point=edge_pos, labware=None)
            pipette.move_to(edge_location)
            protocol.pause(f'Moved to {edge_name} edge')

    # go to bottom last. (If there is more than one well, use the last well first
    # because the pipette is already at the last well at this point)
    for well_loc in reversed(well_locs):
        well = test_labware.well(well_loc)
        set_speeds(RATE)
        pipette.move_to(well.bottom())
        protocol.pause("Moved to the bottom of the well")

        pipette.blow_out(well)

    set_speeds(1.0)
    pipette.return_tip()
