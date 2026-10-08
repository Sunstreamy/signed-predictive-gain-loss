import tempfile
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts')); sys.path.insert(0,str(ROOT/'src'))
from prepare_hai import read_csv
from check_sources import check,digest

class InputPreparation(unittest.TestCase):
    def test_schema_time_labels_and_roundtrip(self):
        columns=['timestamp']+[f'x{i}' for i in range(86)]+['Attack']
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.csv'
            p.write_text(','.join(columns)+'\n'+'2021-01-01 00:00:00,'+','.join(['0.1']*86)+',0\n'+'2021-01-01 00:00:01,'+','.join(['1.2345678901234567']*86)+',1\n')
            x,y,t=read_csv(p,columns)
            np.testing.assert_array_equal(y,[0,1]); self.assertEqual(x.shape,(2,86))
            self.assertEqual(x[1,0],float('1.2345678901234567'))
            with self.assertRaises(ValueError): read_csv(p,['wrong'])
            p.write_text(p.read_text().replace('00:00:01','00:00:03'))
            with self.assertRaises(ValueError): read_csv(p,columns)
    def test_timestamp_reader_crosses_50000_rows(self):
        columns=['timestamp']+[f'x{i}' for i in range(86)]+['Attack']
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'long.csv'
            time=np.datetime64('2021-01-01T00:00:00')+np.arange(50002).astype('timedelta64[s]')
            with p.open('w') as f:
                f.write(','.join(columns)+'\n')
                for t in time: f.write(str(t).replace('T',' ')+','+','.join(['0']*87)+'\n')
            x,y,t=read_csv(p,columns)
            self.assertEqual(len(x),50002); self.assertEqual(len(t),50002)
    def test_hash_not_only_size(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a'; p.write_bytes(b'abc')
            e=dict(name='a',bytes=3,sha256=digest(p))
            self.assertTrue(check([e],d)[0]['verified'])
            p.write_bytes(b'xyz')
            with self.assertRaises(ValueError): check([e],d)

if __name__=='__main__': unittest.main()
