
class AdditiveLossModel:

    @staticmethod
    def get_L_inf(L0, adherence, usage):
        s1 = float(adherence)
        p0 = float(usage)

        assert 0 <= L0
        assert 0 <= s1
        assert 0 <= p0 <= 1
        assert 0 < 1 - s1*p0

        return L0*(1 - p0) / (1 - s1*p0)

    @staticmethod
    def is_will_loop(adherence, usage, A=1):
        s1 = float(adherence)
        p0 = float(usage)

        assert 0 <= s1
        assert 0 <= p0 <= 1

        if s1 == 0:
            return False
        else:
            return s1 < 1 and p0 > A / (1 - s1)

    @staticmethod
    def is_in_loop(k, step, train_size):
        m = step
        n = train_size

        if m == 0:
            return False
        else:
            assert 0 < m <= n
            return 0 <= k <= int(float(n) / m + 1)

    @staticmethod
    def get_p_0(adherence, A=0):
        s1 = float(adherence)
        A = float(A)

        assert 0 <= s1 < 1
        assert 0 <= A <= 1

        return A / (1 - s1)

