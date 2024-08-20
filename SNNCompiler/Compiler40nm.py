from pathlib import Path
import numpy as np
from Common.Common import Fake, div_round_up
from .SMT_Base import SMTBase

class Compiler40nmSNN():
    def __init__(self, config, neuron_num, netbase, network) -> None:
        self.npu_neuron_num = config['Npu_NeuronNum']
        self.chip_npu_num = config['Tile_NpuNum']
        self.npu_num = div_round_up(neuron_num, self.npu_neuron_num)
        self.chip_num = div_round_up(self.npu_num, self.chip_npu_num)

        self.row = config['X_TileNum']
        self.col = config['Y_TileNum']
        self.para_num = min(4, self.col)
        
        self.config = config
        self.neuron_num = neuron_num
        self.netbase = netbase
        self.network = network
        self.asic_flag = self.config['IsAsic']
        self.config['neuron_num'] = self.neuron_num
        self.ndma_staddr = 256 * (16 * (
                self.chip_num - 1) + self.chip_npu_num) if self.asic_flag else 4096 * self.chip_num * self.chip_npu_num

        self.npu_on_s = []
        self.npu_after_s = []
        self.npu_all0_s = []
        self.npu_spike_s = []
        self.spdma_thres = int('0001_000a', 16)
        self.cfg = config['cfg'] << 6

        run, unicast_sel, pkt_tx_en, spike_id_sel, sp_cnt_set, step_set = 0, 0 << 1, 1 << 2, 7 << 3, 0 << 8, 1 << 9
        npu_on_l = self.cfg + run + unicast_sel + pkt_tx_en + spike_id_sel + sp_cnt_set + step_set
        run, unicast_sel, step_set = 1, 1 << 1, 0 << 9
        npu_after_l = self.cfg + run + unicast_sel + pkt_tx_en + spike_id_sel + sp_cnt_set + step_set
        run, unicast_sel = 0, 0 << 1
        npu_all0_l = self.cfg + run + unicast_sel + pkt_tx_en + spike_id_sel + sp_cnt_set + step_set
        run, unicast_sel = 0, 1 << 1
        npu_spike_l = self.cfg + run + unicast_sel + pkt_tx_en + spike_id_sel + sp_cnt_set + step_set

        for j in range(self.chip_num):
            if self.npu_num - self.chip_npu_num * (j) > self.chip_npu_num:
                tmp_s = '1' * self.chip_npu_num
            else:
                tmp_s = '1' * (self.npu_num - self.chip_npu_num * j) + '0' * \
                        (self.chip_npu_num - (self.npu_num - self.chip_npu_num * j))

            value = 0
            for i in range(self.chip_npu_num):
                value = value + int(tmp_s[i]) * (2 ** (self.chip_npu_num - 1 - i))
            value = value << 16
            self.npu_on_s.append(value + npu_on_l)  # 5: 556 7: 572
            self.npu_after_s.append(value + npu_after_l)  # 5: 47 7: 63
            self.npu_all0_s.append(value + npu_all0_l)  # 5: 44 7: 60
            self.npu_spike_s.append(value + npu_spike_l)  # 5: 46 7: 62

        if self.config['cfg'] == 0:
            from .func2smt_fp32 import SMT32bit, SMT96bit      ##FIXME, uger, new version of func2smt.py
            # from .func2smt import SMT32bit, SMT96bit
        elif self.config['cfg'] == 2:
            from .func2smt_int16 import SMT32bit, SMT96bit
        elif self.config['cfg'] == 3:
            from .func2smt_int8 import SMT32bit, SMT96bit

        if self.asic_flag:
            self.smt_compiler = SMT96bit(self.netbase)         ##FIXME, uger, new version of SMT96bit
            # self.smt_compiler = SMT96bit(self.network, self.config)
        else:
            self.smt_compiler = SMT32bit(self.netbase)

    def get_smt_32bit_result(self):
        self.smt_result, self.register_constants = self.smt_compiler.func_to_32bit_smt()

        v_reset = int(np.single(self.netbase.cv['V_reset']).view("uint32").astype("<u4"))
        t_refrac = self.netbase.cv['tau_ref'] + 1
        v_thresh = int(np.single(self.netbase.cv['V_th']).view("uint32").astype("<u4"))
        v_rest = int(np.single(self.netbase.cv['V_rest']).view("uint32").astype("<u4"))

        tw = SMTBase.tw
        step_max = SMTBase.step_max
        neu_nums = SMTBase.neu_num_for32
        rate = SMTBase.rate

        # NOTE: fix the position of register_constants
        shared_property = [0 for _ in range(18)]
        for sr in self.register_constants:
            sr_id = sr.name[2:]  # remove "SR"
            shared_property[int(sr_id)] = int(np.single(sr.value).view("uint32").astype("<u4"))

        self.shared_property_23 = ([self.spdma_thres, v_reset, t_refrac, v_thresh, v_rest]
                                   + shared_property[4:] + [tw, step_max, neu_nums, rate, 16383])

        self.property = []
        for j in range(self.chip_num):
            prop = [j, self.ndma_staddr, self.npu_on_s[j]] + \
                   self.shared_property_23
            self.property.append(prop)

        return self.smt_result, self.property

    def get_smt_96bit_result(self, npu_id):
        self.smt_result = self.smt_compiler.func_to_96bit_smt_cus(npu_id)

        return self.smt_result

    def write_to_bin(self, save_dir):
        """Write hardware related to bin files
        """
        save_dir = Path(save_dir)

        # npu_ctrl
        output_dir = save_dir / 'npu_ctrl'
        output_dir.mkdir(exist_ok=True, parents=True)
        for i in range(self.chip_num):
            file_path = output_dir / f'npu_after_{i}.bin'
            file_path.unlink(missing_ok=True)
            Fake.fwrite(file_path=file_path,
                        arr=self.npu_after_s[i], dtype="<u4")

        for i in range(self.chip_num):
            file_path = output_dir / f'npu_all0_{i}.bin'
            file_path.unlink(missing_ok=True)
            Fake.fwrite(file_path=file_path,
                        arr=self.npu_all0_s[i], dtype="<u4")

        for i in range(self.chip_num):
            file_path = output_dir / f'npu_spike_{i}.bin'
            file_path.unlink(missing_ok=True)
            Fake.fwrite(file_path=file_path,
                        arr=self.npu_spike_s[i], dtype="<u4")

        for i in range(self.chip_num):
            file_path = output_dir / f'npu_on_{i}.bin'
            file_path.unlink(missing_ok=True)
            Fake.fwrite(file_path=file_path, arr=self.npu_on_s[i], dtype="<u4")

        if self.asic_flag:
            # smt 96bit
            output_dir = save_dir / 'smt_96bit'
            output_dir.mkdir(exist_ok=True, parents=True)
            for tile_id in range(self.chip_num):
                # FIXME, not necessary to write hex files
                hex_file_path = output_dir / f'smt.hex'
                bin_file_path = output_dir / f'smt_{tile_id}.bin'
                if bin_file_path.exists():
                    bin_file_path.unlink()

                for npu_id in range(min(self.npu_num, 16)):
                    smt_result = self.get_smt_96bit_result(npu_id)
                    hex_data = []
                    for line in smt_result:
                        instr_bin = ''.join(
                            line.bin_value_for_human.split('_'))
                        ## NOTE: for debug
                        # with open (output_dir / f'smt_binary_line_{tile_id}.txt', 'a') as f_out:
                        #     f_out.write(instr_bin + '\n')
       
                        parts = [instr_bin[i:i + 32]
                                 for i in reversed(range(0, len(instr_bin), 32))]
                        # ['0000021F', '0077F400', '02400000']
                        hex_parts = [format(int(part, 2), '08X')
                                     for part in parts]
                        # ['00000000']
                        padding = ['\\n'.join(['0' * 8])]
                        hex_data.extend(hex_parts + padding)

                    with open(bin_file_path, 'ab') as f_out, open(hex_file_path, 'wt') as f_tmp:
                        for item in hex_data:
                            data_val = int(item, 16)
                            data_arr = np.array([data_val])
                            data_arr.astype("<u4").T.tofile(f_out)
                            f_tmp.write(item + '\n')

            # npu staddr
            output_dir = save_dir / 'ndma_staddr'
            output_dir.mkdir(exist_ok=True, parents=True)
            for i in range(self.chip_num):
                file_path = output_dir / f'ndma_staddr_{i}.bin'
                file_path.unlink(missing_ok=True)
                Fake.fwrite(file_path=file_path, arr=self.ndma_staddr, dtype="<u4")

            # tile id
            output_dir = save_dir / 'tile_id'
            output_dir.mkdir(exist_ok=True, parents=True)
            for i in range(self.chip_num):
                file_path = output_dir / f'tile_id_{i}.bin'
                file_path.unlink(missing_ok=True)
                Fake.fwrite(file_path=file_path, arr=i, dtype="<u4")

            # spdma
            output_dir = save_dir / 'spdma'
            output_dir.mkdir(exist_ok=True, parents=True)
            for i in range(self.chip_num):
                file_path = output_dir / f'spdma_{i}.bin'
                file_path.unlink(missing_ok=True)
                # 655376--'000A_0010'  327688:'5_0008'  131088：‘2_0010’
                Fake.fwrite(file_path=file_path, arr=self.spdma_thres, dtype="<u4")

        else:
            # 32bit hw result
            _ = self.get_smt_32bit_result()

            # property
            output_dir = save_dir / 'property'
            output_dir.mkdir(exist_ok=True, parents=True)
            for i in range(self.chip_num):
                file_path = output_dir / f'property_{i}.bin'
                file_path.unlink(missing_ok=True)
                Fake.fwrite(file_path=file_path,
                            arr=self.property[i], dtype="<u4")

            # smt 32bit
            output_dir = save_dir / 'smt_32bit'
            output_dir.mkdir(exist_ok=True, parents=True)
            instr_bin_all = np.zeros(1024, dtype=np.uint32)
            for n in range(1024):
                if n < len(self.smt_result):
                    instr_bin = ''.join(self.smt_result[n].value)
                    value = int(instr_bin, 2)
                    instr_bin_all[n] = value
            for i in range(self.chip_num):
                file_path = output_dir / f'smt_{i}.bin'
                file_path.unlink(missing_ok=True)
                Fake.fwrite(file_path=file_path,
                            arr=instr_bin_all, dtype="<u4")

            # remote4
            output_dir = save_dir / 'remote4'
            output_dir.mkdir(exist_ok=True, parents=True)
            for gdma_loop in range(1, self.para_num + 1):
                gdma_index = self.col * (gdma_loop - 1)
                for row_index in range(int(self.row / self.para_num)):
                    for column_index in range(self.col):
                        file_path = output_dir / f'remote4_{gdma_index}.bin'
                        file_path.unlink(missing_ok=True)
                        Fake.fwrite(file_path=file_path, arr=[0, gdma_loop], dtype="<u4")
                        gdma_index = gdma_index + 1
                    gdma_index = gdma_index + (self.para_num - 1) * self.col