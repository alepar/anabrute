typedef struct {
    char all_strs[38]; // 36 max
    char offsets[18]; // positives - permutable, negatives - fixed, zeroes - empty; abs(offset)-1 to get offset in all_strs
    ulong start_from;
} permut_template;

ulong fact(uchar x) {
    switch(x) {
        case 0: 	return 1L;
        case 1: 	return 1L;
        case 2: 	return 2L;
        case 3: 	return 6L;
        case 4: 	return 24L;
        case 5: 	return 120L;
        case 6: 	return 720L;
        case 7: 	return 5040L;
        case 8: 	return 40320L;
        case 9: 	return 362880L;
        case 10: 	return 3628800L;
        case 11: 	return 39916800L;
        case 12: 	return 479001600L;
        case 13: 	return 6227020800L;
        case 14: 	return 87178291200L;
        case 15: 	return 1307674368000L;
        case 16: 	return 20922789888000L;
        case 17: 	return 355687428096000L;
        case 18: 	return 6402373705728000L;
        case 19: 	return 121645100408832000L;
        case 20: 	return 2432902008176640000L;
        default:    return 0L;
    }
}

__kernel void permut(__global const permut_template *permut_templates, __global char *permut_str) {
    int id = get_global_id(0);
//    __global const permut_template *tmpl = &permut_templates[id];
    __global const permut_template *tmpl = &permut_templates[0];

    char cur_str[38];
    ulong a = id*16384+1;
    char all_strs[38];
    for (uchar i=0; i<38; i++) {
        all_strs[i] = tmpl->all_strs[i];
    }

    uchar offsets_len;
    char offsets[18];
    for (offsets_len=0; offsets_len<18; offsets_len++) {
        offsets[offsets_len] = tmpl->offsets[offsets_len];
        if (!offsets[offsets_len]) {
            break;
        }
    }

    if (a>1) {
        uchar unpicked_len = 0;
        char unpicked[18];
        uchar idx_unp=0;

        for (uchar i=0; i<offsets_len; i++) {
            if (offsets[i] > 0) {
                unpicked_len++;
                unpicked[idx_unp++] = offsets[i];
            }
        }
        unpicked_len = idx_unp;

        uchar wo=0;
        for (char d=unpicked_len-1; d>=0; d--) {
            ulong factd = fact(d);
            ulong ord = (a-1) / factd;
            a -= ord * factd;

            for (idx_unp=0; idx_unp<unpicked_len; idx_unp++) {
                if (unpicked[idx_unp]>0) {
                    if (ord == 0) {
                        for (; wo<offsets_len; wo++) {
                            if (offsets[wo] > 0) {
                                offsets[wo++] = unpicked[idx_unp];
                                break;
                            }
                        }

                        unpicked[idx_unp] = 0;
                        break;
                    } else {
                        ord--;
                    }
                }
            }
        }
    }

    ushort counter=0;
    do {
        // construct cur_str
        uchar wcs=0;
        for (uchar io=0; io<offsets_len; io++) {
            char off = offsets[io];
            if (off<0) {
                off = -off;
            }
            off--;

            while(all_strs[off]) {
                cur_str[wcs++] = all_strs[off++];
            }
            cur_str[wcs++] = ' ';
        }
        cur_str[wcs-1] = 0;

        // write it out
//        __global char *ds = &permut_str[(id*1024+counter)*38];
//        uchar ic=0;
//        while(cur_str[ic]) {
//            *(ds++) = cur_str[ic++];
//        }

        // find next if possible
        char k = -1;
        char k1 = -1;
        char found = 0;

        for (uchar io=offsets_len-1; io>=0; io--) {
            if (offsets[io]>0) {
                k1 =k;
                k = io;

                if (k1 != -1 && k!=-1 && offsets[k]<offsets[k1]) {
                    found = 1;
                    break;
                }
            }
        }

        if (!found) {
            break;
        }

        uchar l;
        for (l=offsets_len-1; l>k; l--) {
            if (offsets[l]>offsets[k]) {
                break;
            }
        }

        offsets[l] ^= offsets[k];
        offsets[k] ^= offsets[l];
        offsets[l] ^= offsets[k];

        uchar li=k1, ri=offsets_len-1;
        while(li<ri) {
            while(offsets[li]<=0) li++;
            while(offsets[ri]<=0) ri--;

            if (li < ri) {
                offsets[li] ^= offsets[ri];
                offsets[ri] ^= offsets[li];
                offsets[li] ^= offsets[ri];

                li++; ri--;
            }
        }

        counter++;
    } while (counter<16384);

}