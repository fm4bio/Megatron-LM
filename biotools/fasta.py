import hashlib

def read_fasta(fname):
    entries = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                name = line[1:]
                seq = ''
                entries.append((name, seq))
            else:
                entries[-1][1] += line
    return entries

def write_fasta(entries, fname):
    with open(fname, 'w') as f:
        for head, seq in entries:
            f.write(f'>{head}\n')
            f.write(seq + '\n')

def hash_seq(seq):
    return hashlib.md5(seq.encode('utf-8')).hexdigest()