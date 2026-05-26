function hist_rebin = rebin_hist_time(hist_in, factor)

    sz = size(hist_in);
    T = sz(3);

    T_use = floor(T / factor) * factor;
    hist_crop = hist_in(:, :, 1:T_use);

    hist_reshape = reshape(hist_crop, sz(1), sz(2), factor, []);
    hist_rebin = squeeze(sum(hist_reshape, 3));

end