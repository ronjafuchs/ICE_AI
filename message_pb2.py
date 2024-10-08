# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import enum_pb2 as enum__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rmessage.proto\x12\x07service\x1a\nenum.proto\"G\n\x0bGrpcHitArea\x12\x0c\n\x04left\x18\x01 \x01(\x05\x12\r\n\x05right\x18\x02 \x01(\x05\x12\x0b\n\x03top\x18\x03 \x01(\x05\x12\x0e\n\x06\x62ottom\x18\x04 \x01(\x05\"\x9c\x04\n\x0eGrpcAttackData\x12.\n\x10setting_hit_area\x18\x01 \x01(\x0b\x32\x14.service.GrpcHitArea\x12\x17\n\x0fsetting_speed_x\x18\x02 \x01(\x05\x12\x17\n\x0fsetting_speed_y\x18\x03 \x01(\x05\x12.\n\x10\x63urrent_hit_area\x18\x04 \x01(\x0b\x32\x14.service.GrpcHitArea\x12\x15\n\rcurrent_frame\x18\x05 \x01(\x05\x12\x15\n\rplayer_number\x18\x06 \x01(\x08\x12\x0f\n\x07speed_x\x18\x07 \x01(\x05\x12\x0f\n\x07speed_y\x18\x08 \x01(\x05\x12\x10\n\x08start_up\x18\t \x01(\x05\x12\x0e\n\x06\x61\x63tive\x18\n \x01(\x05\x12\x12\n\nhit_damage\x18\x0b \x01(\x05\x12\x14\n\x0cguard_damage\x18\x0c \x01(\x05\x12\x18\n\x10start_add_energy\x18\r \x01(\x05\x12\x16\n\x0ehit_add_energy\x18\x0e \x01(\x05\x12\x18\n\x10guard_add_energy\x18\x0f \x01(\x05\x12\x13\n\x0bgive_energy\x18\x10 \x01(\x05\x12\x10\n\x08impact_x\x18\x11 \x01(\x05\x12\x10\n\x08impact_y\x18\x12 \x01(\x05\x12\x18\n\x10give_guard_recov\x18\x13 \x01(\x05\x12\x13\n\x0b\x61ttack_type\x18\x14 \x01(\x05\x12\x11\n\tdown_prop\x18\x15 \x01(\x08\x12\x15\n\ris_projectile\x18\x16 \x01(\x08\"\x90\x04\n\x11GrpcCharacterData\x12\x15\n\rplayer_number\x18\x01 \x01(\x08\x12\n\n\x02hp\x18\x02 \x01(\x05\x12\x0e\n\x06\x65nergy\x18\x03 \x01(\x05\x12\t\n\x01x\x18\x04 \x01(\x05\x12\t\n\x01y\x18\x05 \x01(\x05\x12\x0c\n\x04left\x18\x06 \x01(\x05\x12\r\n\x05right\x18\x07 \x01(\x05\x12\x0b\n\x03top\x18\x08 \x01(\x05\x12\x0e\n\x06\x62ottom\x18\t \x01(\x05\x12\x0f\n\x07speed_x\x18\n \x01(\x05\x12\x0f\n\x07speed_y\x18\x0b \x01(\x05\x12!\n\x05state\x18\x0c \x01(\x0e\x32\x12.service.GrpcState\x12#\n\x06\x61\x63tion\x18\r \x01(\x0e\x32\x13.service.GrpcAction\x12\r\n\x05\x66ront\x18\x0e \x01(\x08\x12\x0f\n\x07\x63ontrol\x18\x0f \x01(\x08\x12,\n\x0b\x61ttack_data\x18\x10 \x01(\x0b\x32\x17.service.GrpcAttackData\x12\x17\n\x0fremaining_frame\x18\x11 \x01(\x05\x12\x13\n\x0bhit_confirm\x18\x12 \x01(\x08\x12\x16\n\x0egraphic_size_x\x18\x13 \x01(\x05\x12\x16\n\x0egraphic_size_y\x18\x14 \x01(\x05\x12\x18\n\x10graphic_adjust_x\x18\x15 \x01(\x05\x12\x11\n\thit_count\x18\x16 \x01(\x05\x12\x16\n\x0elast_hit_frame\x18\x17 \x01(\x05\x12\x1d\n\x03key\x18\x18 \x01(\x0b\x32\x10.service.GrpcKey\"\xcd\x01\n\rGrpcFrameData\x12\x32\n\x0e\x63haracter_data\x18\x01 \x03(\x0b\x32\x1a.service.GrpcCharacterData\x12\x1c\n\x14\x63urrent_frame_number\x18\x02 \x01(\x05\x12\x15\n\rcurrent_round\x18\x03 \x01(\x05\x12\x30\n\x0fprojectile_data\x18\x04 \x03(\x0b\x32\x17.service.GrpcAttackData\x12\x12\n\nempty_flag\x18\x05 \x01(\x08\x12\r\n\x05\x66ront\x18\x06 \x03(\x08\"J\n\x0bGrpcFftData\x12\x1a\n\x12real_data_as_bytes\x18\x01 \x01(\x0c\x12\x1f\n\x17imaginary_data_as_bytes\x18\x02 \x01(\x0c\"\'\n\x0eGrpcScreenData\x12\x15\n\rdisplay_bytes\x18\x01 \x01(\x0c\"u\n\rGrpcAudioData\x12\x19\n\x11raw_data_as_bytes\x18\x01 \x01(\x0c\x12&\n\x08\x66\x66t_data\x18\x02 \x03(\x0b\x32\x14.service.GrpcFftData\x12!\n\x19spectrogram_data_as_bytes\x18\x03 \x01(\x0c\"`\n\x0cGrpcGameData\x12\x0f\n\x07max_hps\x18\x01 \x03(\x05\x12\x14\n\x0cmax_energies\x18\x02 \x03(\x05\x12\x17\n\x0f\x63haracter_names\x18\x03 \x03(\t\x12\x10\n\x08\x61i_names\x18\x04 \x03(\t\"V\n\x0fGrpcRoundResult\x12\x15\n\rcurrent_round\x18\x01 \x01(\x05\x12\x15\n\rremaining_hps\x18\x02 \x03(\x05\x12\x15\n\relapsed_frame\x18\x03 \x01(\x05\"V\n\x07GrpcKey\x12\t\n\x01\x41\x18\x01 \x01(\x08\x12\t\n\x01\x42\x18\x02 \x01(\x08\x12\t\n\x01\x43\x18\x03 \x01(\x08\x12\t\n\x01U\x18\x04 \x01(\x08\x12\t\n\x01R\x18\x05 \x01(\x08\x12\t\n\x01\x44\x18\x06 \x01(\x08\x12\t\n\x01L\x18\x07 \x01(\x08\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'message_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _GRPCHITAREA._serialized_start=38
  _GRPCHITAREA._serialized_end=109
  _GRPCATTACKDATA._serialized_start=112
  _GRPCATTACKDATA._serialized_end=652
  _GRPCCHARACTERDATA._serialized_start=655
  _GRPCCHARACTERDATA._serialized_end=1183
  _GRPCFRAMEDATA._serialized_start=1186
  _GRPCFRAMEDATA._serialized_end=1391
  _GRPCFFTDATA._serialized_start=1393
  _GRPCFFTDATA._serialized_end=1467
  _GRPCSCREENDATA._serialized_start=1469
  _GRPCSCREENDATA._serialized_end=1508
  _GRPCAUDIODATA._serialized_start=1510
  _GRPCAUDIODATA._serialized_end=1627
  _GRPCGAMEDATA._serialized_start=1629
  _GRPCGAMEDATA._serialized_end=1725
  _GRPCROUNDRESULT._serialized_start=1727
  _GRPCROUNDRESULT._serialized_end=1813
  _GRPCKEY._serialized_start=1815
  _GRPCKEY._serialized_end=1901
# @@protoc_insertion_point(module_scope)
